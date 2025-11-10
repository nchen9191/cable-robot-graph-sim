import math

import torch
import tqdm

from actuation.pid import PID
from gnn_physics.data_processors.batch_tensegrity_data_processor import *
from gnn_physics.data_processors.fast_tensegrity_graph_data_processor import *
from gnn_physics.gnn import *
from robots.tensegrity import TensegrityRobotGNN
from simulators.abstract_simulator import AbstractSimulator
from utilities.tensor_utils import zeros


class TensegrityGNNSimulator(LearnedSimulator):
    def __init__(self,
                 n_out: int,
                 latent_dim: int,
                 nmessage_passing_steps: int,
                 nmlp_layers: int,
                 mlp_hidden_dim: int,
                 tensegrity_cfg,
                 processor_shared_weights=False,
                 dt=0.01,
                 additional_data_proc_kwargs: Dict | None = None):
        data_processor_kwargs = {
            'tensegrity': TensegrityRobotGNN(tensegrity_cfg),
            'dt': dt,
        }
        if additional_data_proc_kwargs is not None:
            data_processor_kwargs.update(additional_data_proc_kwargs)

        super().__init__(n_out,
                         latent_dim,
                         nmessage_passing_steps,
                         nmlp_layers,
                         mlp_hidden_dim,
                         data_processor_kwargs,
                         None,
                         processor_shared_weights)

        self.robot = self.data_processor.robot

        if self.dtype == torch.float64:
            self._encode_process_decode = self._encode_process_decode.double()

        # self.data_processor.compile(fullgraph=True)
        # self._encode_process_decode.compile(fullgraph=True)

    @property
    def num_rods(self):
        return self.robot.num_rods

    def reset(self, **kwargs):
        if 'act_lens' in kwargs:
            cables = self.robot.actuated_cables.values()
            for i, cable in enumerate(cables):
                cable.actuation_length = kwargs['act_lens'][:, i: i + 1].clone()

        if 'motor_speeds' in kwargs:
            cables = self.robot.actuated_cables.values()
            for i, cable in enumerate(cables):
                cable.motor.motor_state.omega_t = kwargs['motor_speeds'][:, i: i + 1].clone()

    def update_state(self, next_state: torch.Tensor) -> None:
        self.robot.update_state(next_state)

    def _get_data_processor(self):
        return BatchTensegrityDataProcessor(**self.data_processor_kwargs)

    def apply_controls(self, control_signals):
        if control_signals is None:
            return

        control_signals = control_signals.reshape(-1, control_signals.shape[1], 1)

        for i in range(control_signals.shape[1]):
            name = f'cable_{i}'
            control = control_signals[:, i: i + 1]

            measure_name = self.robot.cable_map[name]
            measure_cable = self.robot.cables[measure_name]
            cable = self.robot.cables[name]

            curr_length, _ = self.robot.compute_cable_length(measure_cable)
            cable.update_rest_length(control, curr_length, self.dt)

    def _build_gnn(self, **kwargs):
        return EncodeProcessDecode(
            node_types=kwargs['node_types'],
            edge_types=kwargs['edge_types'],
            n_out=kwargs['n_out'],
            latent_dim=kwargs['latent_dim'],
            nmessage_passing_steps=kwargs['nmessage_passing_steps'],
            nmlp_layers=kwargs['nmlp_layers'],
            mlp_hidden_dim=kwargs['mlp_hidden_dim'],
            processor_shared_weights=kwargs['processor_shared_weights'],
            # n_hist=1 + (len(self.prev_states) if self.prev_states else 0)
        )

    def generate_graph(self, state, **kwargs):
        graph = self.data_processor.batch_state_to_graph(state, **kwargs)
        graph = self.data_processor.get_normalize_feats(graph)
        return graph

    def process_gnn(self, graph, **kwargs):
        graph = self._encode_process_decode(graph)

        dv_normalizer = self.data_processor.normalizers['node_dv']

        graph['pf_dv'] = torch.zeros_like(graph.pos)
        graph['p_dv'] = dv_normalizer.inverse(graph['decode_output'])
        graph['p_vel'] = graph.vel + graph.p_dv
        graph['p_pos'] = graph.pos + self.data_processor.dt * graph.p_vel

        return graph

    def step(self,
             curr_state: torch.Tensor,
             ctrls: torch.Tensor | None = None,
             state_to_graph_kwargs: Dict | None = None,
             gnn_kwargs: Dict | None = None,
             **kwargs):
        if state_to_graph_kwargs is None:
            state_to_graph_kwargs = {}
        if gnn_kwargs is None:
            gnn_kwargs = {}

        self.update_state(curr_state)
        self.apply_controls(ctrls)

        graph = self.generate_graph(curr_state, **state_to_graph_kwargs)
        graph = self.process_gnn(graph, **gnn_kwargs)

        body_mask = graph.body_mask.flatten()
        next_state = self.data_processor.node2pose(
            graph.p_pos[body_mask],
            graph.pos[body_mask],
            self.robot.num_nodes_per_rod
        )

        return next_state, graph

    def run(self,
            curr_state: torch.Tensor,
            ctrls: List[torch.Tensor] | torch.Tensor | None,
            num_steps: int = 0,
            state_to_graph_kwargs: List[Dict] | Dict | None = None,
            gnn_kwargs: List[Dict] | Dict | None = None,
            gt_act_lens: torch.Tensor | None = None,
            **kwargs):
        if state_to_graph_kwargs is None:
            state_to_graph_kwargs = {}
        if gnn_kwargs is None:
            gnn_kwargs = {}
        if isinstance(ctrls, list):
            ctrls = torch.cat(ctrls, dim=-1)

        num_steps = len(ctrls) if ctrls is not None else num_steps
        if gt_act_lens is not None:
            assert num_steps == gt_act_lens.shape[-1]

        states, graphs = [], []
        for i in tqdm.tqdm(range(num_steps)):
            s2g_dict = state_to_graph_kwargs[i] \
                if isinstance(state_to_graph_kwargs, list) \
                else state_to_graph_kwargs
            gnn_dict = gnn_kwargs[i] \
                if isinstance(gnn_kwargs, list) \
                else gnn_kwargs

            if gt_act_lens is not None:
                cables = self.robot.actuated_cables.values()
                for j, cable in enumerate(cables):
                    cable.actuation_length = gt_act_lens[:, j, i].reshape(-1, 1, 1)
                controls = None
            else:
                controls = ctrls[..., i: i + 1].clone() if ctrls is not None else None

            curr_state, graph = self.step(
                curr_state,
                controls=controls,
                state_to_graph_kwargs=s2g_dict,
                gnn_kwargs=gnn_dict,
                **kwargs
            )
            states.append(curr_state.clone())
            graphs.append(graph.clone())

        return states, graphs

    def run_target_gait(self, curr_state: torch.Tensor, target_gait, pid_kwargs, **kwargs):
        self.reset()
        self.update_state(curr_state)
        pids = [PID(**pid_kwargs) for _ in range(len(self.robot.actuated_cables))]


class TensegrityMultiSimGNNSimulator(TensegrityGNNSimulator):

    def __init__(self,
                 n_out: int,
                 latent_dim: int,
                 nmessage_passing_steps: int,
                 nmlp_layers: int,
                 mlp_hidden_dim: int,
                 tensegrity_cfg,
                 processor_shared_weights=False,
                 dt=0.01,
                 additional_data_proc_kwargs: Dict | None = None,
                 num_sims=10,
                 cache_batch_sizes=None):
        if additional_data_proc_kwargs is None:
            additional_data_proc_kwargs = {}
        additional_data_proc_kwargs['num_sims'] = num_sims

        if cache_batch_sizes is not None:
            additional_data_proc_kwargs['cache_batch_sizes'] = cache_batch_sizes

        super().__init__(n_out,
                         latent_dim,
                         nmessage_passing_steps,
                         nmlp_layers,
                         mlp_hidden_dim,
                         tensegrity_cfg,
                         processor_shared_weights,
                         dt,
                         additional_data_proc_kwargs)

        # self.data_processor.compile(fullgraph=True)
        # self._encode_process_decode.compile(fullgraph=True)

    def _get_data_processor(self):
        return FastTensegrityGraphDataProcessor(**self.data_processor_kwargs)
        # return MultiSimTensegrityDataProcessor(**self.data_processor_kwargs)

    def generate_graph(self, state, **kwargs):
        batch_size = state.shape[0]
        if batch_size not in self.data_processor.get_cached_batch_size_keys():
            self.data_processor.precompute_and_cache_batch_sizes([batch_size])

        graph = self.data_processor(state, **kwargs)

        return graph

    def process_gnn(self, graph, **kwargs):
        graph_feats, raw_feats = graph

        graph_feats = GraphData(
            **graph_feats._asdict(),
            body_edge_index=graph_feats.body_edge_idx.to(torch.long),
            cable_edge_index=graph_feats.cable_edge_idx.to(torch.long),
            contact_edge_index=graph_feats.contact_edge_idx.to(torch.long),
        )
        mask = graph_feats['contact_close_mask'].flatten()
        graph_feats['contact_edge_index'] = graph_feats['contact_edge_index'][:, mask]
        graph_feats['contact_edge_attr'] = graph_feats['contact_edge_attr'][mask]

        graph_feats = self._encode_process_decode(graph_feats)

        dv_normalizer = self.data_processor.normalizers['node_dv']

        pf_dv = torch.zeros_like(raw_feats[0].node_pos)
        p_dv = dv_normalizer.inverse(graph_feats.decode_output)
        p_vel = raw_feats[0].node_vel + p_dv
        p_pos = raw_feats[0].node_pos + self.data_processor.dt * p_vel

        pred_graph_attrs = PredGnnAttrs(
            pos=raw_feats[0].node_pos,
            vel=raw_feats[0].node_vel,
            p_pos=p_pos,
            p_vel=p_vel,
            pf_dv=pf_dv,
            p_dv=p_dv,
            norm_dv=graph_feats.decode_output,
            body_mask=raw_feats[0].body_mask,
            node_hidden_state=raw_feats[0].node_hidden_state
        )

        return pred_graph_attrs

    # def process_gnn(self, graph, **kwargs):
    #     mask = graph.contact_close_mask.flatten()
    #     graph['contact_edge_index'] = graph['contact_edge_index'][:, mask]
    #     graph['contact_edge_attr'] = graph['contact_edge_attr'][mask]
    #
    #     graph = self.forward(graph)
    #     dv_normalizer = self.data_processor.normalizers['dv']
    #
    #     graph['pf_dv'] = torch.zeros_like(graph.pos)
    #     graph['p_dv'] = dv_normalizer.inverse(graph['decode_output'])
    #     graph['p_vel'] = graph.vel + graph.p_dv
    #     graph['p_pos'] = graph.pos + self.data_processor.dt * graph.p_vel
    #
    #     return graph


class TensegrityRecurrentMotorGNNSimulator(TensegrityGNNSimulator):

    def __init__(self,
                 n_out: int,
                 latent_dim: int,
                 nmessage_passing_steps: int,
                 nmlp_layers: int,
                 mlp_hidden_dim: int,
                 processor_shared_weights=False,
                 dt=0.01,
                 tensegrity_cfg=None,
                 num_ctrls_hist=20,
                 additional_data_proc_kwargs: Dict | None = None):
        self.ctrls_hist = None
        self.node_hidden_state = None
        self.num_ctrls_hist = num_ctrls_hist

        if additional_data_proc_kwargs is None:
            additional_data_proc_kwargs = {}
        additional_data_proc_kwargs['num_ctrls_hist'] = num_ctrls_hist

        super().__init__(
            n_out,
            latent_dim,
            nmessage_passing_steps,
            nmlp_layers,
            mlp_hidden_dim,
            tensegrity_cfg,
            processor_shared_weights,
            dt,
            additional_data_proc_kwargs
        )

    def apply_controls(self, ctrls):
        pass  # do nothing here

    def _get_data_processor(self):
        return BatchTensegrityCtrlsDataProcessor(**self.data_processor_kwargs)

    def reset(self, **kwargs):
        super().reset(**kwargs)
        self.ctrls_hist = kwargs.get('ctrls_hist', None)
        self.node_hidden_state = kwargs.get('node_hidden_state', None)

    def _build_gnn(self, **kwargs):
        return RecurrentMotorEncodeProcessDecode(
            node_types=kwargs['node_types'],
            edge_types=kwargs['edge_types'],
            n_out=kwargs['n_out'],
            latent_dim=kwargs['latent_dim'],
            nmessage_passing_steps=kwargs['nmessage_passing_steps'],
            nmlp_layers=kwargs['nmlp_layers'],
            mlp_hidden_dim=kwargs['mlp_hidden_dim'],
            processor_shared_weights=kwargs['processor_shared_weights'],
        )

    def process_gnn(self, graph, **kwargs):
        graph = self.add_hidden_state(graph)
        graph = self._encode_process_decode(graph)
        dv_normalizer = self.data_processor.normalizers['node_dv']
        cable_normalizer = self.data_processor.normalizers['cable_dl']

        graph['pf_dv'] = torch.zeros_like(graph.pos)
        graph['p_dv'] = dv_normalizer.inverse(graph.decode_output)
        graph['p_vel'] = graph.vel + graph.p_dv
        graph['p_pos'] = graph.pos + self.data_processor.dt * graph.p_vel

        num_cables = len(self.robot.cables)
        cable_dl = graph.cable_decode_output.reshape(-1, 2, 1).transpose(1, 2).reshape(-1, num_cables, 2)
        cable_dl = cable_normalizer.inverse(cable_dl).reshape(-1, 2 * num_cables)
        graph['cable_dl'] = cable_dl

        mean_cable_dl = (cable_dl
                         .reshape(-1, 2)
                         .mean(dim=1, keepdim=True)
                         .reshape(-1, len(self.robot.cables)))

        for i, c in enumerate(self.robot.actuated_cables.values()):
            dl = mean_cable_dl[:, i: i + 1, None]
            c.set_rest_length(c.rest_length - dl)

        self.node_hidden_state = graph['node_hidden_state'].clone()

        return graph

    def add_hidden_state(self, graph):
        for k, v in self.node_types.items():
            if self.node_hidden_state is not None:
                graph[f'{k}_hidden_state'] = self.node_hidden_state
            else:
                graph[f'{k}_hidden_state'] = zeros((
                    graph.node_x.shape[0], 1024),
                    ref_tensor=graph.node_x
                )
        # for k, v in self.edge_types.items():
        #     if k != 'contact' and self.curr_graph is not None and f'{k}_hidden_state' in self.curr_graph:
        #         graph[f'{k}_hidden_state'] = self.curr_graph[f'{k}_hidden_state']

        return graph

    def step(self,
             curr_state: torch.Tensor,
             ctrls: torch.Tensor | None = None,
             state_to_graph_kwargs: Dict | None = None,
             gnn_kwargs: Dict | None = None,
             **kwargs
             ):
        # Save control history
        if self.ctrls_hist is None:
            self.ctrls_hist = zeros(
                (ctrls.shape[0], ctrls.shape[1], self.num_ctrls_hist),  # (B, D, T)
                ref_tensor=ctrls
            )

        assert self.ctrls_hist.shape[0] == ctrls.shape[0], \
            "Controls batch size different than stored control history, need to reset()"

        ctrls = ctrls.reshape(ctrls.shape[0], ctrls.shape[1], 1)
        self.ctrls_hist = torch.concat([self.ctrls_hist[..., 1:], ctrls], dim=2)

        if state_to_graph_kwargs is None:
            state_to_graph_kwargs = {}
        state_to_graph_kwargs['ctrls'] = self.ctrls_hist.clone()

        next_state, graph = super().step(curr_state,
                                         ctrls,
                                         state_to_graph_kwargs,
                                         gnn_kwargs)

        return next_state, graph

    def run(self,
            curr_state: torch.Tensor,
            ctrls: List[torch.Tensor] | torch.Tensor | None,
            num_steps: int = 0,
            state_to_graph_kwargs: List[Dict] | Dict | None = None,
            gnn_kwargs: List[Dict] | Dict | None = None,
            **kwargs):
        if state_to_graph_kwargs is None:
            state_to_graph_kwargs = {}
        if gnn_kwargs is None:
            gnn_kwargs = {}
        if isinstance(ctrls, list):
            ctrls = torch.cat(ctrls, dim=-1)

        num_steps = len(ctrls) if ctrls is not None else num_steps

        states, graphs, rest_lens = [], [], []
        for i in tqdm.tqdm(range(num_steps)):
            s2g_dict = state_to_graph_kwargs[i] \
                if isinstance(state_to_graph_kwargs, list) \
                else state_to_graph_kwargs
            gnn_dict = gnn_kwargs[i] \
                if isinstance(gnn_kwargs, list) \
                else gnn_kwargs

            controls = ctrls[..., i: i + 1].clone()

            curr_state, graph = self.step(
                curr_state,
                controls=controls,
                state_to_graph_kwargs=s2g_dict,
                gnn_kwargs=gnn_dict,
                **kwargs
            )
            states.append(curr_state.clone())
            graphs.append(graph.clone())
            rest_lens.append(torch.hstack([c.rest_length for c in self.robot.actuated_cables.values()]))

        return states, graphs, rest_lens


class TensegrityMultiSimRecurrentMotorGNNSimulator(TensegrityRecurrentMotorGNNSimulator):

    def __init__(self,
                 n_out: int,
                 latent_dim: int,
                 nmessage_passing_steps: int,
                 nmlp_layers: int,
                 mlp_hidden_dim: int,
                 processor_shared_weights=False,
                 dt=0.01,
                 tensegrity_cfg=None,
                 num_sims=10,
                 num_ctrls_hist=20,
                 cache_batch_sizes=None,
                 additional_data_proc_kwargs: Dict | None = None):
        if additional_data_proc_kwargs is None:
            additional_data_proc_kwargs = {}
        additional_data_proc_kwargs['num_sims'] = num_sims
        additional_data_proc_kwargs['num_ctrls_hist'] = num_ctrls_hist
        additional_data_proc_kwargs['cache_batch_sizes'] = cache_batch_sizes

        super().__init__(n_out,
                         latent_dim,
                         nmessage_passing_steps,
                         nmlp_layers,
                         mlp_hidden_dim,
                         processor_shared_weights,
                         dt,
                         tensegrity_cfg,
                         num_ctrls_hist,
                         additional_data_proc_kwargs)
        self.num_sims = num_sims
        # self.data_processor.compile(fullgraph=True)
        # self._encode_process_decode.compile(fullgraph=True)

    def _get_data_processor(self):
        return FastTensegrityGraphDataProcessor(**self.data_processor_kwargs)
        # return MultiSimMotorTensegrityDataProcessor(**self.data_processor_kwargs)

    def generate_graph(self, state, **kwargs):
        batch_size = state.shape[0]
        if batch_size not in self.data_processor.get_cached_batch_size_keys():
            self.data_processor.precompute_and_cache_batch_sizes([batch_size])

        graph_feats, raw_feats = self.data_processor(state, **kwargs)
        combined_graph_feats = {
            **graph_feats._asdict(),
            **{k: v for raw_feat in raw_feats for k, v in raw_feat._asdict().items()},
            'cable_edge_index': graph_feats.cable_edge_idx.to(torch.long),
            'contact_edge_index': graph_feats.contact_edge_idx.to(torch.long),
            'body_edge_index': graph_feats.body_edge_idx.to(torch.long),
            'pos': raw_feats[0].node_pos,
            'vel': raw_feats[0].node_vel,
        }
        graph = GraphData(**combined_graph_feats)
        return graph


class TensegrityRecurrentGNNSimulator(TensegrityGNNSimulator):

    def __init__(self,
                 n_out: int,
                 latent_dim: int,
                 nmessage_passing_steps: int,
                 nmlp_layers: int,
                 mlp_hidden_dim: int,
                 tensegrity_cfg,
                 processor_shared_weights=False,
                 dt=0.01,
                 additional_data_proc_kwargs: Dict | None = None):
        self.node_hidden_state = None
        self.latent_dim = latent_dim

        super().__init__(n_out,
                         latent_dim,
                         nmessage_passing_steps,
                         nmlp_layers,
                         mlp_hidden_dim,
                         tensegrity_cfg,
                         processor_shared_weights,
                         dt,
                         additional_data_proc_kwargs)

        # self._encode_process_decode.compile(full_graph=True)

    def _build_gnn(self, **kwargs):
        return RecurrentEncodeProcessDecode(
            node_types=kwargs['node_types'],
            edge_types=kwargs['edge_types'],
            n_out=kwargs['n_out'],
            latent_dim=kwargs['latent_dim'],
            nmessage_passing_steps=kwargs['nmessage_passing_steps'],
            nmlp_layers=kwargs['nmlp_layers'],
            mlp_hidden_dim=kwargs['mlp_hidden_dim'],
            processor_shared_weights=kwargs['processor_shared_weights'],
        )

    def reset(self, **kwargs):
        super().reset(**kwargs)
        self.node_hidden_state = kwargs.get('node_hidden_state', None)

    def process_gnn(self, graph, **kwargs):
        graph = super().process_gnn(graph, **kwargs)
        self.node_hidden_state = graph.node_hidden_state.clone()

        return graph

    def add_hidden_state(self, graph):
        for k, v in self.node_types.items():
            if self.node_hidden_state is not None:
                graph[f'{k}_hidden_state'] = self.node_hidden_state
            else:
                graph[f'{k}_hidden_state'] = zeros(
                    (graph.node_x.shape[0], 2 * self.latent_dim),
                    ref_tensor=graph.node_x
                )

        return graph


class TensegrityMultiSimRecurrentGNNSimulator(TensegrityRecurrentGNNSimulator):

    def __init__(self,
                 n_out: int,
                 latent_dim: int,
                 nmessage_passing_steps: int,
                 nmlp_layers: int,
                 mlp_hidden_dim: int,
                 tensegrity_cfg,
                 processor_shared_weights=False,
                 dt=0.01,
                 additional_data_proc_kwargs: Dict | None = None,
                 num_sims=10,
                 cache_batch_sizes=None):
        if additional_data_proc_kwargs is None:
            additional_data_proc_kwargs = {}
        additional_data_proc_kwargs['num_sims'] = num_sims
        additional_data_proc_kwargs['cache_batch_sizes'] = cache_batch_sizes

        super().__init__(n_out,
                         latent_dim,
                         nmessage_passing_steps,
                         nmlp_layers,
                         mlp_hidden_dim,
                         tensegrity_cfg,
                         processor_shared_weights,
                         dt,
                         additional_data_proc_kwargs)
        # self.data_processor.compile(fullgraph=True)
        # self._encode_process_decode.compile(fullgraph=True)

    def _get_data_processor(self):
        return FastTensegrityGraphDataProcessor(**self.data_processor_kwargs)
        # return MultiSimTensegrityDataProcessor(**self.data_processor_kwargs)

    def generate_graph(self, state, **kwargs):
        batch_size = state.shape[0]
        if batch_size not in self.data_processor.cached_batch_size_keys:
            self.data_processor.precompute_and_cache_batch_sizes([batch_size])

        graph_feats, raw_feats = self.data_processor(state, **kwargs)
        combined_graph_feats = {
            **graph_feats._asdict(),
            **{k: v for raw_feat in raw_feats for k, v in raw_feat._asdict().items()},
            'cable_edge_index': graph_feats.cable_edge_idx.to(torch.long),
            'contact_edge_index': graph_feats.contact_edge_idx.to(torch.long),
            'body_edge_index': graph_feats.body_edge_idx.to(torch.long),
            'pos': raw_feats[0].node_pos,
            'vel': raw_feats[0].node_vel,
        }
        graph = GraphData(**combined_graph_feats)
        return graph

    def process_gnn(self, graph, **kwargs):
        graph = self.add_hidden_state(graph)
        graph = self._encode_process_decode(graph)

        dv_normalizer = self.data_processor.normalizers['node_dv']

        graph['pf_dv'] = torch.zeros_like(graph.node_pos)
        graph['p_dv'] = dv_normalizer.inverse(graph.decode_output)
        graph['p_vel'] = graph.node_vel + graph.p_dv
        graph['p_pos'] = graph.node_pos + self.data_processor.dt * graph.p_vel

        self.node_hidden_state = graph.node_hidden_state.clone()

        return graph


class MultiSimMultiStepTensegrityGNNSimulator(TensegrityMultiSimGNNSimulator):
    num_out_steps: int

    def __init__(self,
                 n_out: int,
                 latent_dim: int,
                 nmessage_passing_steps: int,
                 nmlp_layers: int,
                 mlp_hidden_dim: int,
                 tensegrity_cfg,
                 processor_shared_weights=False,
                 dt=0.01,
                 additional_data_proc_kwargs: Dict | None = None,
                 num_sims=10,
                 cache_batch_sizes: List[int] | None = None):
        assert n_out % 3 == 0
        self.num_out_steps = n_out // 3

        if additional_data_proc_kwargs is None:
            additional_data_proc_kwargs = {}
        additional_data_proc_kwargs['num_out_steps'] = self.num_out_steps
        additional_data_proc_kwargs['cache_batch_sizes'] = cache_batch_sizes
        additional_data_proc_kwargs['rest_lens_or_ctrls'] = 'rest_lens'

        super().__init__(n_out,
                         latent_dim,
                         nmessage_passing_steps,
                         nmlp_layers,
                         mlp_hidden_dim,
                         tensegrity_cfg,
                         processor_shared_weights,
                         dt,
                         additional_data_proc_kwargs,
                         num_sims)
        self.dt = self.dt.unsqueeze(-1)

    def _get_data_processor(self):
        self.data_processor_kwargs['num_ctrls_hist'] = 0
        return FastTensegrityGraphDataProcessor(**self.data_processor_kwargs)

    def generate_graph(self, state, **kwargs):
        batch_size = state.shape[0]
        if batch_size not in self.data_processor.cached_batch_size_keys:
            self.data_processor.precompute_and_cache_batch_sizes([batch_size])

        graph_feats, raw_feats = self.data_processor(state, **kwargs)
        combined_graph_feats = {
            **graph_feats._asdict(),
            **{k: v for raw_feat in raw_feats for k, v in raw_feat._asdict().items()},
            'cable_edge_index': graph_feats.cable_edge_idx.to(torch.long),
            'contact_edge_index': graph_feats.contact_edge_idx.to(torch.long),
            'body_edge_index': graph_feats.body_edge_idx.to(torch.long),
            'pos': raw_feats[0].node_pos,
            'vel': raw_feats[0].node_vel,
        }
        graph = GraphData(**combined_graph_feats)
        return graph

    def process_gnn(self, graph, **kwargs):
        graph = self._encode_process_decode(graph)

        dv_normalizer = self.data_processor.normalizers['node_dv']
        graph['p_dv'] = dv_normalizer.inverse(graph['decode_output'])
        graph['p_dv'] = graph.p_dv.unsqueeze(1).reshape(graph.p_dv.shape[0], -1, 3).transpose(1, 2)
        graph['p_vel'] = graph.vel.unsqueeze(-1) + torch.cumsum(graph.p_dv, dim=-1)
        graph['p_pos'] = graph.pos.unsqueeze(-1) + torch.cumsum(graph.p_vel * self.dt, dim=-1)

        return graph

    def apply_controls(self, control_signals):
        if control_signals is None:
            return

        all_rest_lens = []
        for i in range(control_signals.shape[-1]):
            ctrls = control_signals[..., i: i + 1]
            super().apply_controls(ctrls)

            rest_lens = torch.hstack([c.rest_length for c in self.robot.actuated_cables.values()])
            all_rest_lens.append(rest_lens)

        self.data_processor.robot.gnn_rest_lens = torch.concat(all_rest_lens, dim=-1)

    def step(self,
             curr_state: torch.Tensor,
             ctrls: torch.Tensor | None = None,
             state_to_graph_kwargs: Dict | None = None,
             gnn_kwargs: Dict | None = None,
             **kwargs):
        if ctrls is not None and ctrls.shape[-1] < self.num_out_steps:
            step_diff = self.num_out_steps - ctrls.shape[-1]
            pad = torch.zeros_like(ctrls[..., :1]).repeat(1, 1, step_diff)
            ctrls = torch.cat([ctrls, pad], dim=-1)
        if state_to_graph_kwargs is None:
            state_to_graph_kwargs = {}
        state_to_graph_kwargs['ctrls'] = ctrls

        return super().step(curr_state,
                            ctrls,
                            state_to_graph_kwargs,
                            gnn_kwargs,
                            **kwargs)

    def run(self,
            curr_state: torch.Tensor,
            ctrls: List[torch.Tensor] | torch.Tensor | None,
            num_steps: int = 0,
            state_to_graph_kwargs: List[Dict] | Dict | None = None,
            gnn_kwargs: List[Dict] | Dict | None = None,
            gt_act_lens: torch.Tensor | None = None,
            show_progress: bool = False,
            **kwargs):
        if state_to_graph_kwargs is None:
            state_to_graph_kwargs = {}
        if gnn_kwargs is None:
            gnn_kwargs = {}
        if isinstance(ctrls, list):
            ctrls = torch.cat(ctrls, dim=-1)

        num_steps = ctrls.shape[-1] if ctrls is not None \
            else gt_act_lens.shape[-1] if gt_act_lens is not None else num_steps
        num_model_steps = math.ceil(round(num_steps / self.num_out_steps, 5))

        states, graphs = [], []

        iterator = range(num_model_steps)
        if show_progress:
            iterator = tqdm.tqdm(iterator)

        for i in iterator:
            start = i * self.num_out_steps
            end = start + self.num_out_steps

            s2g_dict = state_to_graph_kwargs[i] \
                if isinstance(state_to_graph_kwargs, list) \
                else state_to_graph_kwargs
            gnn_dict = gnn_kwargs[i] \
                if isinstance(gnn_kwargs, list) \
                else gnn_kwargs

            if gt_act_lens is not None:
                curr_gt_act_lens = gt_act_lens[..., start:end]
                step_diff = self.num_out_steps - curr_gt_act_lens.shape[-1]
                if step_diff > 0:
                    curr_gt_act_lens = torch.cat([
                        curr_gt_act_lens,
                        curr_gt_act_lens[..., -1:].repeat(1, 1, step_diff),
                    ], dim=-1)

                cables = self.robot.actuated_cables.values()
                cable_init_rest_len = torch.hstack([c._rest_length for c in cables])
                for j, cable in enumerate(cables):
                    cable.actuation_length = curr_gt_act_lens[:, j: j + 1, -1:]
                self.data_processor.robot.gnn_rest_lens = cable_init_rest_len - curr_gt_act_lens
                controls = None
            else:
                controls = ctrls[..., i * self.num_out_steps:(i + 1) * self.num_out_steps] \
                    if ctrls is not None else None

            curr_state, graph = self.step(
                curr_state[..., -1:],
                ctrls=controls,
                state_to_graph_kwargs=s2g_dict,
                gnn_kwargs=gnn_dict,
                **kwargs
            )
            states.extend([curr_state[..., i: i + 1].clone() for i in range(curr_state.shape[-1])])
            graphs.append(graph.clone())

        return states[:num_steps], graphs


class MultiSimMultiStepRecurrentTensegrityGNNSimulator(MultiSimMultiStepTensegrityGNNSimulator):

    def __init__(self,
                 n_out: int,
                 latent_dim: int,
                 nmessage_passing_steps: int,
                 nmlp_layers: int,
                 mlp_hidden_dim: int,
                 tensegrity_cfg,
                 processor_shared_weights=False,
                 dt=0.01,
                 additional_data_proc_kwargs: Dict | None = None,
                 num_sims: int = 10,
                 cache_batch_sizes: List[int] | None = None,
                 torch_compile: bool = False):
        assert n_out % 3 == 0
        self.num_out_steps = n_out // 3
        self.latent_dim = latent_dim

        self.node_hidden_state = None

        if additional_data_proc_kwargs is None:
            additional_data_proc_kwargs = {}
        additional_data_proc_kwargs['cache_batch_sizes'] = cache_batch_sizes
        additional_data_proc_kwargs['recur_latent_dim'] = 2 * self.latent_dim

        super().__init__(n_out,
                         latent_dim,
                         nmessage_passing_steps,
                         nmlp_layers,
                         mlp_hidden_dim,
                         tensegrity_cfg,
                         processor_shared_weights,
                         dt,
                         additional_data_proc_kwargs,
                         num_sims)
        if torch_compile:
            self.run_compile()

    def run_compile(self):
        self.data_processor.compile(fullgraph=True)
        self._encode_process_decode.compile(fullgraph=True)

    def reset(self, **kwargs):
        super().reset(**kwargs)
        self.node_hidden_state = kwargs.get('node_hidden_state', None)

    def _build_gnn(self, **kwargs):
        return RecurrentEncodeProcessDecode(
            node_types=kwargs['node_types'],
            edge_types=kwargs['edge_types'],
            n_out=kwargs['n_out'],
            latent_dim=kwargs['latent_dim'],
            nmessage_passing_steps=kwargs['nmessage_passing_steps'],
            nmlp_layers=kwargs['nmlp_layers'],
            mlp_hidden_dim=kwargs['mlp_hidden_dim'],
            processor_shared_weights=kwargs['processor_shared_weights'],
        )

    def _get_data_processor(self):
        self.data_processor_kwargs['rest_lens_or_ctrls'] = 'rest_lens'
        return FastTensegrityGraphDataProcessor(**self.data_processor_kwargs)

    def add_hidden_state(self, graph):
        if self.node_hidden_state is not None:
            graph[f'node_hidden_state'] = self.node_hidden_state.clone()

        return graph

    def process_gnn(self, graph, **kwargs):
        graph = self.add_hidden_state(graph)
        graph = self._encode_process_decode(graph)

        dv_normalizer = self.data_processor.normalizers['node_dv']
        graph['p_dv'] = dv_normalizer.inverse(graph['decode_output'])
        graph['p_dv'] = graph.p_dv.unsqueeze(1).reshape(graph.p_dv.shape[0], -1, 3).transpose(1, 2)
        graph['p_vel'] = graph.vel.unsqueeze(-1) + torch.cumsum(graph.p_dv, dim=-1)
        graph['p_pos'] = graph.pos.unsqueeze(-1) + torch.cumsum(graph.p_vel * self.dt, dim=-1)

        self.node_hidden_state = graph.node_hidden_state.clone()

        return graph


class MultiSimMultiStepMotorTensegrityGNNSimulator(MultiSimMultiStepTensegrityGNNSimulator):

    def __init__(self,
                 n_out: int,
                 latent_dim: int,
                 nmessage_passing_steps: int,
                 nmlp_layers: int,
                 mlp_hidden_dim: int,
                 tensegrity_cfg,
                 processor_shared_weights=False,
                 dt=0.01,
                 additional_data_proc_kwargs: Dict | None = None,
                 num_sims: int = 10,
                 num_ctrls_hist: int = 20,
                 cache_batch_sizes: List[int] | None = None,
                 torch_compile: bool = False):
        assert n_out % 3 == 0
        self.num_out_steps = n_out // 3
        self.num_ctrls_hist = num_ctrls_hist
        self.num_ctrls_per_cable = num_ctrls_hist + self.num_out_steps

        self.latent_dim = latent_dim

        self.ctrls_hist = None
        self.node_hidden_state = None

        if additional_data_proc_kwargs is None:
            additional_data_proc_kwargs = {}
        additional_data_proc_kwargs['cache_batch_sizes'] = cache_batch_sizes
        additional_data_proc_kwargs['recur_latent_dim'] = 2 * self.latent_dim

        super().__init__(n_out,
                         latent_dim,
                         nmessage_passing_steps,
                         nmlp_layers,
                         mlp_hidden_dim,
                         tensegrity_cfg,
                         processor_shared_weights,
                         dt,
                         additional_data_proc_kwargs,
                         num_sims)
        if torch_compile:
            self.run_compile()

    def run_compile(self):
        self.data_processor.compile(fullgraph=True)
        self._encode_process_decode.compile(fullgraph=True)

    def reset(self, **kwargs):
        super().reset(**kwargs)
        self.ctrls_hist = kwargs.get('ctrls_hist', None)
        self.node_hidden_state = kwargs.get('node_hidden_state', None)

    def _build_gnn(self, **kwargs):
        # return RecurrentMotorEncodeProcessDecode(
        return MotorEncodeProcessDecode(
            node_types=kwargs['node_types'],
            edge_types=kwargs['edge_types'],
            n_out=kwargs['n_out'],
            latent_dim=kwargs['latent_dim'],
            nmessage_passing_steps=kwargs['nmessage_passing_steps'],
            nmlp_layers=kwargs['nmlp_layers'],
            mlp_hidden_dim=kwargs['mlp_hidden_dim'],
            processor_shared_weights=kwargs['processor_shared_weights'],
        )

    def _get_data_processor(self):
        self.data_processor_kwargs['rest_lens_or_ctrls'] = 'ctrls'
        self.data_processor_kwargs['num_ctrls_hist'] = self.num_ctrls_hist
        return FastTensegrityGraphDataProcessor(**self.data_processor_kwargs)

    def generate_graph(self, state, **kwargs):
        graph = super().generate_graph(state, **kwargs)
        graph = self.add_hidden_state(graph)
        return graph

    def process_gnn(self, graph, **kwargs):
        graph = super().process_gnn(graph, **kwargs)

        num_steps = kwargs.get('num_steps', self.num_out_steps)

        cable_normalizer = self.data_processor.normalizers['cable_dl']
        cable_actuated_mask = graph['cable_actuated_mask'].flatten()
        graph['act_cable_dl'] = cable_normalizer.inverse(graph['cable_decode_output'][cable_actuated_mask])
        mean_act_cable_dl = torch.cumsum(
            graph['act_cable_dl']
            .reshape(-1, 2, self.num_out_steps)
            .mean(dim=1, keepdim=True)
            .reshape(-1, len(self.robot.actuated_cables), self.num_out_steps),
            dim=2
        )
        curr_rest_lens = torch.hstack([c.rest_length for c in self.robot.actuated_cables.values()])
        graph['next_rest_lens'] = curr_rest_lens - mean_act_cable_dl

        for i, c in enumerate(self.robot.actuated_cables.values()):
            c.set_rest_length(graph['next_rest_lens'][:, i: i + 1, num_steps - 1: num_steps])

        return graph

    def apply_controls(self, control_signals):
        pass  # do nothing

    def add_hidden_state(self, graph):
        if self.node_hidden_state is not None:
            graph[f'node_hidden_state'] = self.node_hidden_state.clone()

        return graph

    def step(self,
             curr_state: torch.Tensor,
             ctrls: torch.Tensor | None = None,
             state_to_graph_kwargs: Dict | None = None,
             gnn_kwargs: Dict | None = None,
             **kwargs):
        if self.ctrls_hist is None:
            self.ctrls_hist = torch.zeros_like(ctrls[..., :1]).repeat(1, 1, self.num_ctrls_hist)

        ctrls = torch.concat([self.ctrls_hist, ctrls], dim=2)
        self.ctrls_hist = ctrls[..., -self.num_ctrls_hist:].clone()

        act_cables = list(self.robot.actuated_cables.values())
        if act_cables[0].rest_length.shape[0] == 1:
            for c in act_cables:
                rest_len = c.rest_length.repeat(ctrls.shape[0], 1, 1)
                c.set_rest_length(rest_len)

        next_state, next_graph = super().step(curr_state,
                                              ctrls,
                                              state_to_graph_kwargs,
                                              gnn_kwargs,
                                              **kwargs)

        self.node_hidden_state = next_graph['node_hidden_state'].clone()

        return next_state, next_graph

    def run(self,
            curr_state: torch.Tensor,
            ctrls: List[torch.Tensor] | torch.Tensor | None,
            num_steps: int = 0,
            state_to_graph_kwargs: List[Dict] | Dict | None = None,
            gnn_kwargs: List[Dict] | Dict | None = None,
            gt_act_lens: torch.Tensor | None = None,
            show_progress=False,
            **kwargs):
        if state_to_graph_kwargs is None:
            state_to_graph_kwargs = {}
        if gnn_kwargs is None:
            gnn_kwargs = {}
        if isinstance(ctrls, list):
            ctrls = torch.cat(ctrls, dim=-1)

        num_steps = ctrls.shape[-1] if ctrls is not None else num_steps
        if gt_act_lens is not None:
            assert num_steps == gt_act_lens.shape[-1]
        num_multi_steps = math.ceil(round(num_steps / self.num_out_steps, 5))
        iterator = range(num_multi_steps)
        if show_progress:
            iterator = tqdm.tqdm(iterator)

        states, graphs, rest_lens = [], [], []
        for i in iterator:
            start = i * self.num_out_steps
            n = min(num_steps - start, self.num_out_steps)
            end = start + n

            s2g_dict = state_to_graph_kwargs[i] \
                if isinstance(state_to_graph_kwargs, list) \
                else state_to_graph_kwargs
            gnn_dict = gnn_kwargs[i] \
                if isinstance(gnn_kwargs, list) \
                else gnn_kwargs
            gnn_dict['num_steps'] = n

            if gt_act_lens is not None:
                cables = self.robot.actuated_cables.values()
                cable_init_rest_len = torch.hstack([c._rest_length for c in cables])
                for j, cable in enumerate(cables):
                    cable.actuation_length = gt_act_lens[:, j: j + 1, end - 1: end]
                self.data_processor.robot.gnn_rest_lens = cable_init_rest_len - gt_act_lens[..., start:end]
                controls = None
            else:
                controls = ctrls[..., start:end] \
                    if ctrls is not None else None

            if controls is not None:
                step_diff = self.num_out_steps - controls.shape[-1]
                if step_diff > 0:
                    pad = torch.zeros_like(controls[..., :1]).repeat(1, 1, step_diff)
                    controls = torch.cat([controls, pad], dim=-1)

            curr_state, graph = self.step(
                curr_state[..., -1:],
                ctrls=controls,
                state_to_graph_kwargs=s2g_dict,
                gnn_kwargs=gnn_dict,
                **kwargs
            )

            graphs.append(graph.clone())
            states.extend([
                curr_state[..., i: i + 1].clone()
                for i in range(controls.shape[-1])
            ])
            rest_lens.extend([
                graph['next_rest_lens'][..., i: i + 1].clone()
                for i in range(controls.shape[-1])
            ])

        return states[:ctrls.shape[-1]], graphs, rest_lens[:ctrls.shape[-1]]

    # def run_target_gait(self,
    #                     curr_state: torch.Tensor,
    #                     state_to_graph_kwargs: List[Dict] | Dict | None = None,
    #                     gnn_kwargs: List[Dict] | Dict | None = None,
    #                     **kwargs):
    #     if state_to_graph_kwargs is None:
    #         state_to_graph_kwargs = {}
    #     if gnn_kwargs is None:
    #         gnn_kwargs = {}
    #
    #     pids = [PID(min_length=0.9, RANGE=1.1, tol=0.1) for _ in range(self.robot.actuated_cables)]
    #
    #     ctrls = torch.ones(
    #         (curr_state.shape[0], len(self.robot.actuated_cables), 1),
    #         device=curr_state.device
    #     )
    #     step = 0
    #     while (ctrls != 0.0).any() or step < 1000:
    #         for cable in self.robot.actuated_cables.values():


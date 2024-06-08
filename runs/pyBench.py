import numpy as np
import matplotlib.pyplot as plt
import pprint
pp = pprint.PrettyPrinter(indent=2)



targets = ['matmul_blas', 'matmul_shmem', 'matmul_lanes', 'stencil_blas', 'stencil_lanes']

def load():
    raw_data = dict()
    DATA_LEN = 7200
    for target in targets:
        reps, Ns, numRHSs = [np.zeros(DATA_LEN, dtype=np.int32) for _ in range(3)]
        grids = np.zeros(DATA_LEN, dtype=np.dtype('<U11'))
        times = np.zeros((DATA_LEN, 4), dtype=np.int32)
        with open(target+'.out', 'r') as file:
            for i, raw_line in enumerate(file.readlines()[1:]):
                line = raw_line.strip().split(',')
                reps[i] = int(line[0])
                grids[i] = line[1]
                Ns[i] = int(line[2])
                numRHSs[i] = int(line[3])
                if target == 'matmul_blas':
                    times[i, 1] = int(line[5])
                    times[i, 2] = int(line[6])
                    times[i, 3] = int(line[7])
                elif target == 'stencil_blas':
                    times[i, 0] = int(line[5])
                    times[i, 1] = int(line[6])
                    times[i, 2] = int(line[7])
                    times[i, 3] = int(line[8])
                else:
                    times[i, 2] = int(line[5])
        raw_data[target] = dict()
        raw_data[target]['rep']     = np.array(reps)
        raw_data[target]['grid']    = np.array(grids)
        raw_data[target]['N']       = np.array(Ns)
        raw_data[target]['numRHS']  = np.array(numRHSs)
        raw_data[target]['time']    = np.array(times)
    return raw_data

class Plotter:
    targets  = ['matmul_blas', 'matmul_shmem', 'matmul_lanes', 'stencil_blas', 'stencil_lanes']
    Ns       = [32, 64, 128]
    numRHSs  = [1, 12, 24, 36, 48, 60]
    grids    = ['4.4.4.4', '4.4.8.8', '8.8.8.8', '16.16.16.16']
    n_reps   = 100
    data_len = 7200
    data_len_reduced = data_len // 100
    n_time_slices = 4
    n_targets = len(targets)
    slice_labels = ['malloc', 'cp_in', 'op', 'cp_out']

    def __init__(self, data: dict):
        self.data = data

    @staticmethod
    def get_indices_lower_median(data: np.ndarray):
        return np.argwhere(data <= np.median(data))[0:len(data)//2]

    def reduce(self):
        self.reduced_data = dict()
        for target in targets:
            reduced_data_target = dict()

            # parameters
            reduced_data_target['grid'] = self.data[target]['grid'][::Plotter.n_reps]
            reduced_data_target['N'] = self.data[target]['N'][::Plotter.n_reps]
            reduced_data_target['numRHS'] = self.data[target]['numRHS'][::Plotter.n_reps]

            # reducing the times
            summed_times = np.sum(self.data[target]['time'], axis=1)
            summed_times_resized = np.resize(summed_times, new_shape=(Plotter.data_len_reduced, Plotter.n_reps))

            reduced_data_target['time'] = np.zeros(shape = (Plotter.data_len_reduced, Plotter.n_time_slices)
                                                 , dtype = np.float64)
            times_reshaped = np.resize(self.data[target]['time'], new_shape=(Plotter.data_len_reduced, 
                                                                             Plotter.n_reps, 
                                                                             Plotter.n_time_slices))

            for i in range(Plotter.data_len_reduced):
                indices = Plotter.get_indices_lower_median(summed_times_resized[i])
                time_slice = np.mean(times_reshaped[i][indices], axis = 0)
                reduced_data_target['time'][i] = time_slice

            self.reduced_data[target] = reduced_data_target
        
    def get_reduced_time_slices(self, target: str, N: int, numRHS: int, grid: str):
        N_mask = self.reduced_data['matmul_blas']['N'] == N
        numRHS_mask = self.reduced_data['matmul_blas']['numRHS'] == numRHS
        grid_mask = self.reduced_data['matmul_blas']['grid'] == grid
        index = np.argwhere(np.all([N_mask, numRHS_mask, grid_mask], axis=0))[0]
        return self.reduced_data[target]['time'][index]

    def plot_fixed_grid(self, grid: str = '8.8.8.8'):
        fig, axs = plt.subplots(nrows=len(Plotter.Ns), ncols=len(Plotter.numRHSs)
                                , sharex=True, sharey=True
                                , figsize=(12,6))

        cols_labels = [f"numRHS = {str(numRHS)}" for numRHS in Plotter.numRHSs] 
        rows_labels = [f"N = {str(N)}\ntime in us" for N in Plotter.Ns] 
        for ax, col in zip(axs[0], cols_labels):
            ax.set_title(col)
        for ax, row in zip(axs[:,0], rows_labels):
            ax.set_ylabel(row)#, rotation=0, size='large')

        for N, axs_fixed_N in zip(Plotter.Ns, axs):
            for numRHS, ax in zip(Plotter.numRHSs, axs_fixed_N):
                slice_data = np.zeros((Plotter.n_targets, Plotter.n_time_slices), dtype=np.float64)
                for i, target in enumerate(Plotter.targets):
                    slice_data[i] = self.get_reduced_time_slices(target, N, numRHS, grid)
                slice_data = slice_data.T
                bottom = np.zeros(Plotter.n_targets, dtype=np.float64)
                for i, slice_label in enumerate(Plotter.slice_labels):
                    ax.bar(Plotter.targets, slice_data[i], label=slice_label, bottom=bottom)
                    bottom += slice_data[i]
                ax.tick_params(axis='x', labelrotation=90)

        axs[0][-1].legend()

        fig.suptitle(f"BENCHMARKS: TYPE=realF, GRID={grid}")
        fig.tight_layout()
        plt.show()

if __name__ == "__main__":
    data = load()
    plotter = Plotter(data)
    plotter.reduce()
    pp.pprint(plotter.reduced_data)
    plotter.plot_fixed_grid()


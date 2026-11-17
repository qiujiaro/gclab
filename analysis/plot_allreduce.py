"""Plot M2 ring/tree crossover and chunk-size sweeps."""


def load_results(csv_path):
    # TODO(M2): load CSV, invoke schema.validate_rows, and return a typed table.
    del csv_path
    raise NotImplementedError("TODO(M2): load_results")


def plot_crossover(table, output_path):
    # TODO(M2): plot ring vs tree by message size, separating topology/rank
    # count and visibly labelling every series SIMULATED.
    del table, output_path
    raise NotImplementedError("TODO(M2): plot_crossover")


def plot_chunk_sweep(table, output_path):
    # TODO(M2): plot latency and busbw versus chunk size.
    del table, output_path
    raise NotImplementedError("TODO(M2): plot_chunk_sweep")

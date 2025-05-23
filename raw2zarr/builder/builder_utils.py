import icechunk


def get_icechunk_session(zarr_store: str, branch: str = "main"):
    storage = icechunk.local_filesystem_storage(zarr_store)
    try:
        repo = icechunk.Repository.create(storage)
    except icechunk.IcechunkError:
        repo = icechunk.Repository.open(storage)
    return repo.writable_session(branch)

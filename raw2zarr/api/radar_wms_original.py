# radar_wms_api.py

import numpy as np
import xarray as xr

import wradlib as wrl
from fastapi import FastAPI, Query, HTTPException
from fastapi.responses import StreamingResponse
from io import BytesIO
import xradar as xd
from pyproj import CRS, Transformer
import matplotlib
from zarr.testing.strategies import zarr_formats

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def get_geocoords(ds) -> xr.Dataset:
    """Attach georeferenced coordinates to a radar sweep."""
    ds = ds.xradar.georeference()

    src_crs = ds.xradar.get_crs()
    trg_crs = CRS.from_epsg(4326)
    transformer = Transformer.from_crs(src_crs, trg_crs)

    trg_y, trg_x, trg_z = transformer.transform(ds.x, ds.y, ds.z)
    ds = ds.assign_coords(
        {
            "lon": (ds.x.dims, trg_x, xd.model.get_longitude_attrs()),
            "lat": (ds.y.dims, trg_y, xd.model.get_latitude_attrs()),
            "alt": (ds.z.dims, trg_z, xd.model.get_altitude_attrs()),
        }
    )
    return ds


# ⚠️ Load one sweep from the NEXRAD file and georeference
dtree = xd.io.open_nexradlevel2_datatree(
    "/media/alfonso/drive/Alfonso/python/raw2zarr/data/KVNX20110520_000023_V06"
)
dtree = xr.open_datatree(
    "/media/alfonso/drive/Alfonso/python/data/tw-data.zarr",
    zarr_format=3,
    consolidated=False,
)
# dtree = dtree.xradar.map_over_sweeps(get_geocoords)
# ds = dtree["sweep_0"].ds
ds = dtree["denton.tx/sweep_0"].ds
app = FastAPI()


def render_wms_tile(
    ds, var="DBZH", bbox=(-98.3, 36.5, -97.9, 37.0), width=512, height=512
):
    """Interpolate radar data to a lat/lon grid and return PNG image."""

    if var not in ds:
        raise HTTPException(
            status_code=400, detail=f"Variable '{var}' not found in dataset."
        )

    proj = ds.xradar.get_crs()

    # Transform BBOX to projection coordinates
    transformer = Transformer.from_crs("EPSG:4326", proj, always_xy=True)
    xmin, ymin = transformer.transform(bbox[0], bbox[1])
    xmax, ymax = transformer.transform(bbox[2], bbox[3])

    # Create target projected grid
    x = np.linspace(xmin, xmax, width)
    y = np.linspace(ymin, ymax, height)
    cart = xr.Dataset(coords={"x": ("x", x), "y": ("y", y)})

    center = (float(ds.y.mean().item()), float(ds.x.mean().item()))
    gridded_ds = ds.wrl.comp.togrid(
        cart, radius=450000.0, center=center, interpol=wrl.ipol.Nearest
    )

    grid = gridded_ds[var].values

    # ✅ Transform grid corners BACK to EPSG:4326 for accurate plotting
    inv_transformer = Transformer.from_crs(proj, "EPSG:4326", always_xy=True)
    lon_min, lat_min = inv_transformer.transform(xmin, ymin)
    lon_max, lat_max = inv_transformer.transform(xmax, ymax)
    # Plot with matplotlib
    aspect_ratio = height / width
    fig_width = 6  # inches
    fig_height = fig_width * aspect_ratio
    fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=100)
    ax.imshow(
        grid,
        origin="lower",
        extent=(lon_min, lon_max, lat_min, lat_max),
        cmap="ChaseSpectral",
        vmin=-10,
        vmax=70,
    )
    ax.axis("off")
    plt.tight_layout()

    buf = BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    buf.seek(0)

    return StreamingResponse(buf, media_type="image/png")


@app.get("/wms")
def get_wms_tile(
    var: str = "DBZH",
    bbox: str = Query("-98.3,36.5,-97.9,37.0"),
    width: int = 512,
    height: int = 512,
):
    bbox_tuple = tuple(map(float, bbox.split(",")))
    return render_wms_tile(ds, var=var, bbox=bbox_tuple, width=width, height=height)

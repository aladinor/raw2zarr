#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import xradar as xd
import pandas as pd
from matplotlib.animation import FuncAnimation
import cartopy.crs as ccrs
from pandas import to_datetime
import matplotlib.pyplot as plt


def plt_ppi(ds):
    fig, ax = plt.subplots()
    ds.sel(times='2023-04-07 03:20', method='nearest').DBZH.plot(x="x", y='y', vmin=-10, vmax=50,
                                                                 cmap="Spectral_r", )
    m2km = lambda x, _: f"{x / 1000:g}"
    # set new ticks
    ax.xaxis.set_major_formatter(m2km)
    ax.yaxis.set_major_formatter(m2km)
    ax.set_ylabel("$North - South \ distance \ [km]$")
    ax.set_xlabel("$East - West \ distance \ [km]$")
    ax.set_title(
        f"$Guaviare \ radar$"
        + "\n"
        + f"${to_datetime(ds.sel(times='2023-04-07 03:20', method='nearest').times.values): %Y-%m-%d - %H:%M}$"
        + "$ UTC$"
    )
    ax.set_ylim(-300000, 300000)
    ax.set_xlim(-300000, 300000)
    plt.show()


def plot_anim(ds, save=False):
    fig, ax = plt.subplots(subplot_kw={"projection": ccrs.PlateCarree()})
    proj_crs = xd.georeference.get_crs(ds)
    cart_crs = ccrs.Projection(proj_crs)
    sc = ds.isel(times=0).DBZH.plot.pcolormesh(x="x", y="y", vmin=-10,
                                               vmax=50, cmap="Spectral_r",
                                               transform=cart_crs,
                                               ax=ax)

    title = f"Barranca radar - {ds.isel(times=0).sweep_fixed_angle.values: .1f} [deg] \n " \
            f"{pd.to_datetime(ds.isel(times=0).time.values[0]):%Y-%m-%d %H:%M}UTC"
    ax.set_title(title)
    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                      linewidth=1, color="gray", alpha=0.3, linestyle="--")
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    gl.top_labels = False
    gl.right_labels = False
    ax.coastlines()

    def update_plot(t):
        sc.set_array(ds.sel(times=t).DBZH.values.ravel())
        ax.set_title(f"Barranca radar - {ds.sel(times=t).sweep_fixed_angle.values: .1f} [deg] \n "
                     f"{pd.to_datetime(ds.sel(times=t).time.values[0]):%Y-%m-%d %H:%M}UTC")

    ani = FuncAnimation(fig, update_plot, frames=ds.times.values,
                        interval=5)
    plt.show()

    if save:
        writervideo = ani.FFMpegWriter(fps=60)
        ani.save(f'/media/alfonso/drive/Alfonso/zarr_radar/ani.mp4')
        print(1)
        pass


import numpy as np
import matplotlib.pyplot as plt
import NuRadioReco.modules.io.NuRadioRecoio
from NuRadioReco.framework.parameters import showerParameters as shp
from NuRadioReco.framework.parameters import stationParameters as stnp
from NuRadioReco.framework.parameters import channelParameters as chp
import glob

io = NuRadioReco.modules.io.NuRadioRecoio.NuRadioRecoio(glob.glob('../electric_field_reconstruction/data/*.nur'))


channel_groups = [
    [0, 1, 2, 3, 7, 8],
    [9, 10, 11],
    [12, 13, 14],
    [4],
    [5],
    [6]
]
hor_distances = np.arange(100., 3900., 4.)
distance_uncertainties = np.zeros((io.get_n_events(), 2))
sim_distances = np.zeros(io.get_n_events())
rec_distances = np.zeros(io.get_n_events())
channel_group_snrs = np.zeros((io.get_n_events(), len(channel_groups)))
corr_cuts = [.98, .99]
snr_cut = 2.5

for i_event, event in enumerate(io.get_events()):
    if i_event >= 500000:
        break
    station = event.get_station(101)
    sim_vertex = None
    for sim_shower in event.get_sim_showers():
        sim_vertex = sim_shower.get_parameter(shp.vertex)
    sim_distances[i_event] = np.sqrt(sim_vertex[0]**2 + sim_vertex[1]**2)
    rec_vertex = station.get_parameter(stnp.nu_vertex)
    rec_distances[i_event] = np.sqrt(rec_vertex[0]**2 + rec_vertex[1]**2)
    distance_correlation = np.zeros_like(hor_distances)
    correlations = station.get_parameter(stnp.distance_correlations)
    distance_correlation[:len(correlations)] = correlations / np.max(correlations)
    distance_uncertainties[i_event, 0] = np.min(hor_distances[distance_correlation > corr_cuts[0]])
    distance_uncertainties[i_event, 1] = np.max(hor_distances[distance_correlation > corr_cuts[1]])
    for i_group, channel_group in enumerate(channel_groups):
        max_group_snr = 0
        for channel_id in channel_group:
            channel = station.get_channel(channel_id)
            if channel.has_parameter(chp.signal_region_snrs):
                ch_snr = channel.get_parameter(chp.signal_region_snrs)
                if len(ch_snr) > 0:
                    if np.max(ch_snr) > max_group_snr:
                        max_group_snr = np.max(ch_snr)
        channel_group_snrs[i_event, i_group] = max_group_snr

bad_event_mask = np.max(channel_group_snrs, axis=1) < snr_cut
bottom_only_mask = (np.max(channel_group_snrs[:, :3], axis=1) >= snr_cut) & (np.max(channel_group_snrs[:, 3:], axis=1) < snr_cut)
bottom_plus_1_mask = (np.max(channel_group_snrs[:, :3], axis=1) >= snr_cut) & (np.sum(channel_group_snrs[:, 3:] >= snr_cut, axis=1) >= 1)

fig1 = plt.figure()
ax1_1 = fig1.add_subplot(111)
ax1_1.errorbar(
    sim_distances[bottom_plus_1_mask],
    rec_distances[bottom_plus_1_mask],
    yerr=[
        (rec_distances - distance_uncertainties[:, 0])[bottom_plus_1_mask],
        (distance_uncertainties[:, 1] - rec_distances)[bottom_plus_1_mask]
    ],
    fmt='o'
)
ax1_1.plot(
    [0, 4000],
    [0, 4000],
    color='k'
)
ax1_1.set_aspect('equal')
ax1_1.grid()
fig1.tight_layout()
fig1.savefig('plots/uncertainties/scatter.png')

fig2 = plt.figure(figsize=(8, 12))
ax2_1 = fig2.add_subplot(211)
ax2_1.scatter(
    (np.abs(rec_distances - sim_distances))[bad_event_mask],
    ((distance_uncertainties[:, 1] - distance_uncertainties[:, 0]))[bad_event_mask],
    alpha=.1,
    color='k'
)
ax2_1.scatter(
    (np.abs(rec_distances - sim_distances))[bottom_only_mask],
    ((distance_uncertainties[:, 1] - distance_uncertainties[:, 0]))[bottom_only_mask],
    alpha=.3,
    color='C0'
)
ax2_1.scatter(
    (np.abs(rec_distances - sim_distances))[bottom_plus_1_mask],
    ((distance_uncertainties[:, 1] - distance_uncertainties[:, 0]))[bottom_plus_1_mask],
    alpha=.3,
    color='C1'
)
ax2_1.plot(
    [.1, 10000],
    [.1, 10000],
    color='k'
)
ax2_1.grid()
ax2_1.set_xscale('log')
ax2_1.set_yscale('log')
ax2_1.set_xlim([1, 5000])
ax2_1.set_ylim([1, 5000])
ax2_2 = fig2.add_subplot(212)
ax2_2.scatter(
    (np.abs(rec_distances - sim_distances))[bottom_plus_1_mask],
    (np.abs(distance_uncertainties[:, 1] - sim_distances))[bottom_plus_1_mask],
    alpha=.3,
    color='C0'
)
ax2_2.scatter(
    (np.abs(rec_distances - sim_distances))[bottom_plus_1_mask],
    (np.abs(sim_distances - distance_uncertainties[:, 0]))[bottom_plus_1_mask],
    alpha=.3,
    color='C1'
)
ax2_2.plot(
    [.001, 10000],
    [.001, 10000],
    color='k'
)
ax2_2.grid()
ax2_2.set_xscale('log')
ax2_2.set_yscale('log')
ax2_2.set_xlim([.1, 5000])
ax2_2.set_ylim([.1, 5000])
fig2.tight_layout()
fig2.savefig('plots/uncertainties/error_uncertainties_scatter.png')
import numpy as np
import matplotlib.pyplot as plt
import NuRadioReco.modules.io.NuRadioRecoio
from NuRadioReco.utilities import units
from NuRadioReco.framework.parameters import channelParameters as chp
from NuRadioReco.framework.parameters import stationParameters as stnp
from NuRadioReco.framework.parameters import showerParameters as shp
import glob

io = NuRadioReco.modules.io.NuRadioRecoio.NuRadioRecoio(glob.glob('../electric_field_reconstruction/data/*.nur'))
snr_cut = 2.5
attenuation_length = 947. * units.km
channel_groups = [
    [0, 1, 2, 3],
    [9, 10],
    [12, 13],
    [4],
    [5],
    [6]
]
rec_vertices = np.zeros((io.get_n_events(), 3))
sim_vertices = np.zeros((io.get_n_events(), 3))
channel_group_snrs = np.zeros((io.get_n_events(), len(channel_groups)))
flavors = np.zeros(io.get_n_events())
interaction_types = np.empty(io.get_n_events(), dtype=str)
energies = np.zeros(io.get_n_events())
for i_event, event in enumerate(io.get_events()):
    station = event.get_station(101)
    for sim_shower in event.get_sim_showers():
        sim_vertices[i_event] = sim_shower.get_parameter(shp.vertex)
        flavors[i_event] = sim_shower.get_parameter(shp.flavor)
        interaction_types[i_event] = sim_shower.get_parameter(shp.interaction_type)
        energies[i_event] += sim_shower.get_parameter(shp.energy)
    rec_vertices[i_event] = station.get_parameter(stnp.nu_vertex)
    for i_group, channel_group in enumerate(channel_groups):
        for channel_id in channel_group:
            channel = station.get_channel(channel_id)
            if channel.has_parameter(chp.signal_region_snrs):
                if len(channel.get_parameter(chp.signal_region_snrs)) > 0:
                    max_snr = np.max(channel.get_parameter(chp.signal_region_snrs))
                    if max_snr > channel_group_snrs[i_event, i_group]:
                        channel_group_snrs[i_event, i_group] = max_snr

rec_distances = np.sqrt(np.sum(rec_vertices**2, axis=1))
sim_distances = np.sqrt(np.sum(sim_vertices**2, axis=1))
sim_attenuation = np.exp(-sim_distances / attenuation_length) / (sim_distances / units.km)
rec_attenuation = np.exp(-rec_distances / attenuation_length) / (rec_distances / units.km)
quality_class_mask = np.zeros((5, io.get_n_events()), dtype=bool)
quality_class_mask[0] = (np.max(channel_group_snrs, axis=1) < snr_cut)
quality_class_mask[1] = (np.max(channel_group_snrs[:, :3], axis=1) >= snr_cut) & (np.max(channel_group_snrs[:, 3:], axis=1) < snr_cut)
quality_class_mask[2] = (np.max(channel_group_snrs[:, :3], axis=1) >= snr_cut) & (np.sum(channel_group_snrs[:, 3:] >= snr_cut, axis=1) == 1)
quality_class_mask[3] = (np.max(channel_group_snrs[:, :3], axis=1) >= snr_cut) & (np.sum(channel_group_snrs[:, 3:] >= snr_cut, axis=1) == 2)
quality_class_mask[4] = (np.max(channel_group_snrs[:, :3], axis=1) >= snr_cut) & (np.sum(channel_group_snrs[:, 3:] >= snr_cut, axis=1) == 3)

fig1 = plt.figure(figsize=(8, 12))
distance_error_bins = np.arange(-1500, 1501, 100)
distance_errors_clipped = np.clip(rec_distances - sim_distances, a_min=distance_error_bins[0], a_max=distance_error_bins[-1])
ax1_1 = fig1.add_subplot(311)
ax1_1.hist(
    [
        # distance_errors_clipped[quality_class_mask[0]],
        distance_errors_clipped[quality_class_mask[1]],
        distance_errors_clipped[quality_class_mask[2]],
        distance_errors_clipped[quality_class_mask[3]],
        distance_errors_clipped[quality_class_mask[4]]
    ],
    color=['C0', 'C1', 'C2', 'C3'],
    bins=distance_error_bins,
    edgecolor='k',
    stacked=True,
    label=[
        'bottom only',
        'bottom + 1',
        'bottom + 2',
        'bottom + 3'
    ]
)
ax1_1.set_xlabel(r'$d_{rec} - d_{sim}$ [m]')
ax1_1.grid()
ax1_1.legend()
ax1_2 = fig1.add_subplot(312)
rel_distance_error_bins = np.arange(-1, 2. + 1.e-6, .1)
rel_distance_errors_clipped = np.clip(
    (rec_distances - sim_distances) / sim_distances,
    a_min=rel_distance_error_bins[0],
    a_max=rel_distance_error_bins[-1]
)
ax1_2.hist(
    [
        # rel_distance_errors_clipped[quality_class_mask[0]],
        rel_distance_errors_clipped[quality_class_mask[1]],
        rel_distance_errors_clipped[quality_class_mask[2]],
        rel_distance_errors_clipped[quality_class_mask[3]],
        rel_distance_errors_clipped[quality_class_mask[4]]
    ],
    color=['C0', 'C1', 'C2', 'C3'],
    bins=rel_distance_error_bins,
    edgecolor='k',
    stacked=True
)
ax1_2.grid()
ax1_2.set_xlabel(r'$(d_{rec} - d_{sim}) / d_{sim}$')
ax1_3 = fig1.add_subplot(313)
attenuation_bins = np.power(10., np.arange(-1., 1., .05))
attenuation_errors_clippes = np.clip(rec_attenuation / sim_attenuation, a_min=attenuation_bins[0], a_max=attenuation_bins[-1])
ax1_3.hist(
    [
        # attenuation_errors_clippes[quality_class_mask[0]],
        attenuation_errors_clippes[quality_class_mask[1]],
        attenuation_errors_clippes[quality_class_mask[2]],
        attenuation_errors_clippes[quality_class_mask[3]],
        attenuation_errors_clippes[quality_class_mask[4]]
    ],
    color=['C0', 'C1', 'C2', 'C3'],
    bins=attenuation_bins,
    edgecolor='k',
    stacked=True
)
ax1_3.set_xscale('log')
ax1_3.grid()
ax1_3.set_xlabel(r'$(e^{- d_{rec}/l_{att}} / d_{rec}) / (e^{- d_{sim}/l_{att}}/d_{sim})$')
fig1.tight_layout()
fig1.savefig('plots/reco_quality/distance_error_hist.png')

fig2 = plt.figure(figsize=(8, 8))
ax2_1 = fig2.add_subplot(111)
ax2_1.grid()
ax2_1.set_aspect('equal')
energy_mask = (energies < 1.e22)
ax2_1.scatter(
    np.sqrt(sim_vertices[:, 0]**2 + sim_vertices[:, 1]**2)[quality_class_mask[0] & energy_mask],
    sim_vertices[:, 2][quality_class_mask[0] & energy_mask],
    c='gray'
)
ax2_1.scatter(
    np.sqrt(sim_vertices[:, 0]**2 + sim_vertices[:, 1]**2)[quality_class_mask[1] & energy_mask],
    sim_vertices[:, 2][quality_class_mask[1] & energy_mask],
    c='C0'
)
ax2_1.scatter(
    np.sqrt(sim_vertices[:, 0]**2 + sim_vertices[:, 1]**2)[quality_class_mask[2] & energy_mask],
    sim_vertices[:, 2][quality_class_mask[2] & energy_mask],
    c='C1'
)
ax2_1.scatter(
    np.sqrt(sim_vertices[:, 0]**2 + sim_vertices[:, 1]**2)[quality_class_mask[3] & energy_mask],
    sim_vertices[:, 2][quality_class_mask[3] & energy_mask],
    c='C2'
)
ax2_1.scatter(
    np.sqrt(sim_vertices[:, 0]**2 + sim_vertices[:, 1]**2)[quality_class_mask[4] & energy_mask],
    sim_vertices[:, 2][quality_class_mask[4] & energy_mask],
    c='C3'
)
ax2_1.axvline(
    100,
    color='k',
    linestyle='--'
)
fig2.tight_layout()
fig2.savefig('plots/reco_quality/event_geometries.png')

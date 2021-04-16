import numpy as np
import glob
import matplotlib.pyplot as plt
import matplotlib.colors
from NuRadioReco.utilities import units, trace_utilities
from NuRadioReco.framework.parameters import electricFieldParameters as efp
from NuRadioReco.framework.parameters import showerParameters as shp
from NuRadioReco.framework.parameters import stationParameters as stnp
from NuRadioReco.framework.parameters import channelParameters as chp
import NuRadioReco.framework.base_trace
import NuRadioReco.modules.io.NuRadioRecoio
import NuRadioReco.detector.generic_detector
import NuRadioMC.SignalProp.analyticraytracing
import NuRadioMC.utilities.medium
import pickle

io = NuRadioReco.modules.io.NuRadioRecoio.NuRadioRecoio(glob.glob('../electric_field_reconstruction/data/*.nur'))

det = NuRadioReco.detector.generic_detector.GenericDetector(
    json_filename='RNO_station.json',
    default_station=101,
    default_channel=0
)
snr_cutoff = 2.5
passbands = [
    [
        [.13, .2],
        [.2, .3]
    ],
    [
        [.13, .3],
        [.3, .5]
    ]
]
channel_groups = [
    [0, 1, 2, 3, 7, 8],
    [9, 10, 11],
    [12, 13, 14],
    [4],
    [5],
    [6]
]
cmap = plt.get_cmap('viridis', 8)
fit_parameter_pickle = pickle.load(open('../spectrum_shape_parametrization/fit_parameters.p', 'rb'))

ice = NuRadioMC.utilities.medium.get_ice_model('greenland_simple')
sim_energies = np.zeros(io.get_n_events())
rec_energies = np.zeros((io.get_n_events(), len(channel_groups)))
best_rec_energies = np.zeros(io.get_n_events())
rec_energies[:] = np.nan
interaction_types = np.empty(io.get_n_events(), dtype=str)
nu_flavors = np.zeros(io.get_n_events(), dtype=int)
empty_event_filter = np.ones(io.get_n_events(), dtype=bool)
max_snrs = np.zeros((io.get_n_events(), len(channel_groups)))

raytracer = NuRadioMC.SignalProp.analyticraytracing.ray_tracing(
    ice,
    'GL1'
)
for i_event, event in enumerate(io.get_events()):
    if i_event % 50 == 0:
        print('Event {}/{}'.format(i_event, io.get_n_events()))
    station = event.get_station(101)
    sim_station = station.get_sim_station()
    for sim_shower in event.get_sim_showers():
        sim_energies[i_event] += sim_shower.get_parameter(shp.energy)
        nu_flavors[i_event] = sim_shower.get_parameter(shp.flavor)
        interaction_types[i_event] = sim_shower.get_parameter(shp.interaction_type)
    rec_vertex = station.get_parameter(stnp.nu_vertex)
    raytracer.set_start_and_end_point(rec_vertex, det.get_relative_position(101, 0))
    raytracer.find_solutions()
    for i_group, channel_group in enumerate(channel_groups):
        max_snr_in_group = 0
        for efield in station.get_electric_fields_for_channels(channel_group):
            ray_type = efield.get_parameter(efp.ray_path_type)
            channel = station.get_channel(channel_group[0])
            # Check if this is the ray type with the largest SNR
            if not channel.has_parameter(chp.signal_region_snrs) or not channel.has_parameter(chp.signal_ray_types):
                continue
            current_snr = 0
            for i_region, region_snr in enumerate(channel.get_parameter(chp.signal_region_snrs)):
                if ray_type == channel.get_parameter(chp.signal_ray_types)[i_region]:
                    current_snr = region_snr
            if current_snr <= max_snr_in_group:
                continue
            max_snr_in_group = current_snr
            attenuation = None
            path_length = None
            for i_solution in range(raytracer.get_number_of_raytracing_solutions()):
                if raytracer.get_solution_type(i_solution) == ray_type:
                    attenuation = raytracer.get_attenuation(
                        i_solution,
                        efield.get_frequencies(),
                        10. * units.GHz
                    )
                    path_length = raytracer.get_path_length(i_solution)
            if attenuation is not None:
                corrected_efield = NuRadioReco.framework.base_trace.BaseTrace()
                corrected_efield.set_frequency_spectrum(
                    efield.get_frequency_spectrum() / attenuation * (path_length / units.km),
                    efield.get_sampling_rate()
                )
                energy_fluence_1 = trace_utilities.get_electric_field_energy_fluence(
                    corrected_efield.get_filtered_trace(passbands[0][0], 'butter', 10),
                    corrected_efield.get_times()
                )
                energy_fluence_2 = trace_utilities.get_electric_field_energy_fluence(
                    corrected_efield.get_filtered_trace(passbands[0][1], 'butter', 10),
                    corrected_efield.get_times()
                )
                fit_parameters = fit_parameter_pickle['{:.0f}-{:.0f}/{:.0f}-{:.0f}'.format(
                    passbands[0][0][0] / units.MHz,
                    passbands[0][0][1] / units.MHz,
                    passbands[0][1][0] / units.MHz,
                    passbands[0][1][1] / units.MHz
                )]
                if energy_fluence_1[1] / energy_fluence_2[1] > 10:
                    energy_fluence_1 = trace_utilities.get_electric_field_energy_fluence(
                        corrected_efield.get_filtered_trace(passbands[1][0], 'butter', 10),
                        corrected_efield.get_times()
                    )
                    energy_fluence_2 = trace_utilities.get_electric_field_energy_fluence(
                        corrected_efield.get_filtered_trace(passbands[1][1], 'butter', 10),
                        corrected_efield.get_times()
                    )
                    fit_parameters = fit_parameter_pickle['{:.0f}-{:.0f}/{:.0f}-{:.0f}'.format(
                        passbands[1][0][0] / units.MHz,
                        passbands[1][0][1] / units.MHz,
                        passbands[1][1][0] / units.MHz,
                        passbands[1][1][1] / units.MHz
                    )]
                log_s_parameter = np.log10((energy_fluence_1[1]) / (energy_fluence_2[1]))
                rec_energy = np.sqrt(np.sum(energy_fluence_1)) / (
                        fit_parameters[0] * log_s_parameter**2 + fit_parameters[1] * log_s_parameter + fit_parameters[2]
                ) * units.EeV
                rec_energies[i_event, i_group] = rec_energy
    if np.sum(np.isnan(rec_energies[i_event])) > 0:
        empty_event_filter[i_event] = False
    for i_group, channel_group in enumerate(channel_groups):
        max_snr = 0
        for channel_id in channel_group:
            channel = station.get_channel(channel_id)
            noise_rms = channel.get_parameter(chp.noise_rms)
            sim_channel_sum = None
            for sim_channel in sim_station.get_channels_by_channel_id(channel_id):
                if sim_channel_sum is None:
                    sim_channel_sum = sim_channel
                else:
                    sim_channel_sum += sim_channel
            if sim_channel_sum is not None:
                snr = .5 * (np.max(sim_channel_sum.get_trace()) - np.min(
                    sim_channel_sum.get_trace())) / noise_rms
                if max_snr < snr:
                    max_snrs[i_event, i_group] = snr
                    max_snr = snr

em_shower_mask = (np.abs(nu_flavors) == 12) & (interaction_types == 'c')
bottom_channel_mask = np.sum(~np.isnan(rec_energies[:, :3]), axis=1) > 0
signal_channel_mask = (np.max(max_snrs[:, :3], axis=1) >= 2.5) & (np.sum(max_snrs[:, 3:] >= snr_cutoff, axis=1) >= 2)
weak_signal_channel_mask = (np.max(max_snrs[:, :3], axis=1) >= 2.5) & (np.sum(max_snrs[:, 3:] >= snr_cutoff, axis=1) >= 1)

rec_energy_means = np.zeros(io.get_n_events())
# rec_energy_means[np.isnan(rec_energies[:, 0])] = np.nanmean(rec_energies, axis=1)[np.isnan(rec_energies[:, 0])]
rec_energy_means[bottom_channel_mask] = np.nanmean(rec_energies[:, :3], axis=1)[bottom_channel_mask]
rec_energy_means[~bottom_channel_mask] = np.nanmean(rec_energies, axis=1)[~bottom_channel_mask]
rec_energy_means[~np.isnan(rec_energies[:, 0])] = rec_energies[:, 0][~np.isnan(rec_energies[:, 0])]
fig1 = plt.figure(figsize=(8, 6))
ax1_1 = fig1.add_subplot(111)
ax1_1.grid()
ax1_1.set_xscale('log')
ax1_1.set_yscale('log')
rec_energy_scatter = ax1_1.scatter(
    sim_energies[signal_channel_mask & ~em_shower_mask],
    rec_energy_means[signal_channel_mask & ~em_shower_mask],
    marker='o',
    c=np.max(max_snrs[:, :3], axis=1)[signal_channel_mask & ~em_shower_mask],
    cmap=cmap,
    vmin=2,
    vmax=10,
)
ax1_1.scatter(
    sim_energies[~signal_channel_mask & ~em_shower_mask & weak_signal_channel_mask],
    rec_energy_means[~signal_channel_mask & ~em_shower_mask & weak_signal_channel_mask],
    marker='o',
    c=np.max(max_snrs, axis=1)[~signal_channel_mask & ~em_shower_mask & weak_signal_channel_mask],
    alpha=.3,
    cmap=cmap,
    vmin=2,
    vmax=10,
)
ax1_1.scatter(
    sim_energies[signal_channel_mask & em_shower_mask],
    rec_energy_means[signal_channel_mask & em_shower_mask],
    marker='x',
    c=np.max(max_snrs, axis=1)[signal_channel_mask & em_shower_mask],
    cmap=cmap,
    vmin=2,
    vmax=10
)
ax1_1.scatter(
    sim_energies[~signal_channel_mask & em_shower_mask & weak_signal_channel_mask],
    rec_energy_means[~signal_channel_mask & em_shower_mask & weak_signal_channel_mask],
    marker='x',
    c=np.max(max_snrs, axis=1)[~signal_channel_mask & em_shower_mask & weak_signal_channel_mask],
    alpha=.3,
    cmap=cmap,
    vmin=2,
    vmax=10
)
ax1_1.scatter(
    sim_energies[~weak_signal_channel_mask & ~em_shower_mask],
    rec_energy_means[~weak_signal_channel_mask & ~em_shower_mask],
    marker='o',
    c='k',
    alpha=.2,
    cmap=cmap,
    vmin=2,
    vmax=10
)
ax1_1.scatter(
    sim_energies[~weak_signal_channel_mask & em_shower_mask],
    rec_energy_means[~weak_signal_channel_mask & em_shower_mask],
    marker='x',
    c='k',
    alpha=.2,
    cmap=cmap,
    vmin=2,
    vmax=10
)
plt.colorbar(rec_energy_scatter, ax=ax1_1).set_label('SNR')
ax1_1.set_xlim([1.e16, 1.e19])
ax1_1.set_ylim([1.e16, 1.e19])
x_points = np.power(10, np.arange(15, 20, .5))
ax1_1.plot(x_points, x_points, color='k')
ax1_1.plot(x_points, 2. * x_points, color='k', linestyle=':')
ax1_1.plot(x_points, .5 * x_points, color='k', linestyle=':')
ax1_1.set_xlabel(r'$E_{sh}^{sim}$')
ax1_1.set_ylabel(r'$E_{sh}^{rec}$')
ax1_1.set_aspect('equal')
fig1.tight_layout()
fig1.savefig('plots/rec_energy_scatter.png')

fig2 = plt.figure(figsize=(8, 12))
ax2_1 = fig2.add_subplot(312)
ax2_1.grid()
ax2_1.set_xscale('log')
energy_bins = np.power(10, np.arange(-2, 2, .05))
ax2_1.hist(
    np.clip((rec_energy_means / sim_energies)[~em_shower_mask], a_min=energy_bins[0], a_max=energy_bins[-1]),
    bins=energy_bins,
    edgecolor='k'
)
ax2_1.hist(
    np.clip((rec_energy_means / sim_energies)[weak_signal_channel_mask & ~em_shower_mask], a_min=energy_bins[0], a_max=energy_bins[-1]),
    bins=energy_bins,
    edgecolor='k'
)
ax2_1.hist(
    np.clip((rec_energy_means / sim_energies)[signal_channel_mask & ~em_shower_mask], a_min=energy_bins[0], a_max=energy_bins[-1]),
    bins=energy_bins,
    edgecolor='k'
)
ax2_1.set_title('hadronic showers')
ax2_1.set_xlabel('$E_{sh}^{rec} / E_{sh}^{sim}$')
ax2_2 = fig2.add_subplot(313)
ax2_2.grid()
ax2_2.set_xscale('log')
ax2_2.hist(
    np.clip((rec_energy_means / sim_energies)[em_shower_mask], a_min=energy_bins[0], a_max=energy_bins[-1]),
    bins=energy_bins,
    edgecolor='k'
)
ax2_2.hist(
    np.clip((rec_energy_means / sim_energies)[weak_signal_channel_mask & em_shower_mask], a_min=energy_bins[0], a_max=energy_bins[-1]),
    bins=energy_bins,
    edgecolor='k'
)
ax2_2.hist(
    np.clip((rec_energy_means / sim_energies)[signal_channel_mask & em_shower_mask], a_min=energy_bins[0], a_max=energy_bins[-1]),
    bins=energy_bins,
    edgecolor='k'
)

ax2_2.set_title(r'$\nu_e + cc$')
ax2_2.set_xlabel('$E_{sh}^{rec} / E_{sh}^{sim}$')

ax2_3 = fig2.add_subplot(311)
ax2_3.grid()
ax2_3.set_xscale('log')
ax2_3.hist(
    np.clip(rec_energy_means / sim_energies, a_min=energy_bins[0], a_max=energy_bins[-1]),
    bins=energy_bins,
    edgecolor='k'
)
ax2_3.hist(
    np.clip((rec_energy_means / sim_energies)[weak_signal_channel_mask], a_min=energy_bins[0], a_max=energy_bins[-1]),
    bins=energy_bins,
    edgecolor='k'
)
ax2_3.hist(
    np.clip((rec_energy_means / sim_energies)[signal_channel_mask], a_min=energy_bins[0], a_max=energy_bins[-1]),
    bins=energy_bins,
    edgecolor='k'
)
ax2_3.set_title('all events')
ax2_3.set_xlabel('$E_{sh}^{rec} / E_{sh}^{sim}$')
fig2.tight_layout()
fig2.savefig('plots/rec_energy_hist.png')

energy_bin_size = .5
log_energy_bins = np.arange(16.5, 19.5, energy_bin_size)
energy_percentiles = np.zeros((len(log_energy_bins), 3, 2, 3))
for i_bin, log_energy_bin in enumerate(log_energy_bins):
    energy_filter = (np.log10(sim_energies) > log_energy_bin) & (np.log10(sim_energies) < log_energy_bin + energy_bin_size)
    if len(rec_energies[signal_channel_mask & energy_filter]) > 0:
        energy_percentiles[i_bin, 0, 0] = np.percentile(
            (rec_energy_means / sim_energies)[signal_channel_mask & energy_filter],
            q=[16, 84, 50]
        )
    if len(rec_energies[weak_signal_channel_mask & energy_filter]) > 0:
        energy_percentiles[i_bin, 0, 1] = np.percentile(
            (rec_energy_means / sim_energies)[weak_signal_channel_mask & energy_filter],
            q=[16, 84, 50]
        )
    if len(rec_energies[signal_channel_mask & energy_filter & ~em_shower_mask]) > 0:
        energy_percentiles[i_bin, 1, 0] = np.percentile(
            (rec_energy_means / sim_energies)[signal_channel_mask & energy_filter & ~em_shower_mask],
            q=[16, 84, 50]
        )
    if len(rec_energies[weak_signal_channel_mask & energy_filter & ~em_shower_mask]) > 0:
        energy_percentiles[i_bin, 1, 1] = np.percentile(
            (rec_energy_means / sim_energies)[weak_signal_channel_mask & energy_filter & ~em_shower_mask],
            q=[16, 84, 50]
        )
    if len(rec_energies[signal_channel_mask & energy_filter & em_shower_mask]) > 0:
        energy_percentiles[i_bin, 2, 0] = np.percentile(
            (rec_energy_means / sim_energies)[signal_channel_mask & energy_filter & em_shower_mask],
            q=[16, 84, 50]
        )
    if len(rec_energies[weak_signal_channel_mask & energy_filter & em_shower_mask]) > 0:
        energy_percentiles[i_bin, 2, 1] = np.percentile(
            (rec_energy_means / sim_energies)[weak_signal_channel_mask & energy_filter & em_shower_mask],
            q=[16, 84, 50]
        )
energy_percentiles[energy_percentiles <= 0] = np.nan
fig3 = plt.figure(figsize=(6, 10))
ax3_1 = fig3.add_subplot(211)
ax3_1.grid()
ax3_1.set_xscale('log')
ax3_1.plot(
    np.power(10, log_energy_bins),
    energy_percentiles[:, 0, 0, 2],
    color='C0',
    label='all'
)
ax3_1.fill_between(
    np.power(10, log_energy_bins),
    energy_percentiles[:, 0, 0, 0],
    energy_percentiles[:, 0, 0, 1],
    facecolor=matplotlib.colors.to_rgba('C0', .3),
    edgecolor=matplotlib.colors.to_rgba('C0', 1.),
    step='post',
    linestyle='-',
    linewidth=2
)
ax3_1.plot(
    np.power(10, log_energy_bins),
    energy_percentiles[:, 1, 0, 2],
    color='C1',
    label='hadronic showers',
    linestyle='--'
)
ax3_1.fill_between(
    np.power(10, log_energy_bins),
    energy_percentiles[:, 1, 0, 0],
    energy_percentiles[:, 1, 0, 1],
    facecolor=matplotlib.colors.to_rgba('C1', .3),
    edgecolor=matplotlib.colors.to_rgba('C1', 1.),
    step='post',
    linestyle='--',
    linewidth=2
)
ax3_1.plot(
    np.power(10, log_energy_bins),
    energy_percentiles[:, 2, 0, 2],
    color='C2',
    label=r'$~nu_e + cc$',
    linestyle=':'
)
ax3_1.fill_between(
    np.power(10, log_energy_bins),
    energy_percentiles[:, 2, 0, 0],
    energy_percentiles[:, 2, 0, 1],
    facecolor=matplotlib.colors.to_rgba('C2', .3),
    edgecolor=matplotlib.colors.to_rgba('C2', 1.),
    step='post',
    linestyle=':',
    linewidth=2
)
ax3_1.set_xlabel(r'$E_{sh}^{sim}$ [eV]')
ax3_1.set_ylabel(r'$E_{sh}^{rec} / E_{sh}^{sim}$')
ax3_1.legend()
ax3_2 = fig3.add_subplot(212, sharey=ax3_1)
ax3_2.grid()
ax3_2.set_xscale('log')
ax3_2.plot(
    np.power(10, log_energy_bins),
    energy_percentiles[:, 0, 1, 2],
    color='C0',
    label='all'
)
ax3_2.fill_between(
    np.power(10, log_energy_bins),
    energy_percentiles[:, 0, 1, 0],
    energy_percentiles[:, 0, 1, 1],
    facecolor=matplotlib.colors.to_rgba('C0', .3),
    edgecolor=matplotlib.colors.to_rgba('C0', 1.),
    step='post',
    linestyle='-',
    linewidth=2
)
ax3_2.plot(
    np.power(10, log_energy_bins),
    energy_percentiles[:, 1, 1, 2],
    color='C1',
    label='hadronic showers'
)
ax3_2.fill_between(
    np.power(10, log_energy_bins),
    energy_percentiles[:, 1, 1, 0],
    energy_percentiles[:, 1, 1, 1],
    facecolor=matplotlib.colors.to_rgba('C1', .3),
    edgecolor=matplotlib.colors.to_rgba('C1', 1.),
    step='post',
    linestyle='--',
    linewidth=2
)
ax3_2.plot(
    np.power(10, log_energy_bins),
    energy_percentiles[:, 2, 1, 2],
    color='C2',
    label=r'$~nu_e + cc$'
)
ax3_2.fill_between(
    np.power(10, log_energy_bins),
    energy_percentiles[:, 2, 1, 0],
    energy_percentiles[:, 2, 1, 1],
    facecolor=matplotlib.colors.to_rgba('C2', .3),
    edgecolor=matplotlib.colors.to_rgba('C2', 1.),
    step='post',
    linestyle=':',
    linewidth=2
)
ax3_2.set_xlabel(r'$E_{sh}^{sim}$ [eV]')
ax3_2.set_ylabel(r'$E_{sh}^{rec} / E_{sh}^{sim}$')
ax3_2.legend()
ax3_2.set_ylim([0, 2])
fig3.tight_layout()
fig3.savefig('plots/reconstruction_quantiles.png')

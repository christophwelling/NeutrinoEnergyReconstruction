import numpy as np
import glob
import matplotlib.pyplot as plt
import NuRadioReco.modules.io.NuRadioRecoio
from NuRadioReco.utilities import units, trace_utilities
from NuRadioReco.framework.parameters import electricFieldParameters as efp
from NuRadioReco.framework.parameters import channelParameters as chp
import pickle

event_reader = NuRadioReco.modules.io.NuRadioRecoio.NuRadioRecoio(glob.glob('data/*.nur'))
ray_types = ['direct', 'refracted', 'reflected']
cmap = plt.get_cmap('inferno', 5)
fit_parameter_pickle = pickle.load(open('../spectrum_shape_parametrization/fit_parameters.p', 'rb'))

channel_groups = [
    [0, 1, 2, 3, 7, 8],
    [9, 10, 11],
    [12, 13, 14],
    [4],
    [5],
    [6]
]
passbands = [
    [.13, .3],
    [.3, .5]
]
fit_parameters = fit_parameter_pickle['{:.0f}-{:.0f}/{:.0f}-{:.0f}'.format(
    passbands[0][0] / units.MHz,
    passbands[0][1] / units.MHz,
    passbands[1][0] / units.MHz,
    passbands[1][1] / units.MHz
)]
rec_energy_fluences = np.zeros((event_reader.get_n_events(), len(channel_groups), 3, 3))
sim_energy_fluences = np.zeros((event_reader.get_n_events(), len(channel_groups), 3, 3))
sim_slopes = np.zeros((event_reader.get_n_events(), len(channel_groups), 3))
rec_slopes = np.zeros((event_reader.get_n_events(), len(channel_groups), 3))
has_rec_efield = np.zeros((event_reader.get_n_events(), len(channel_groups), 3), dtype=bool)
has_sim_efield = np.zeros((event_reader.get_n_events(), len(channel_groups), 3), dtype=bool)
max_snrs = np.zeros((event_reader.get_n_events(), len(channel_groups)))
rec_energy_parameters = np.zeros((event_reader.get_n_events(), len(channel_groups), 3))
sim_energy_parameters = np.zeros((event_reader.get_n_events(), len(channel_groups), 3))


for i_event, event in enumerate(event_reader.get_events()):
    station = event.get_station(101)
    sim_station = station.get_sim_station()
    for i_group, channel_group in enumerate(channel_groups):
        for efield in station.get_electric_fields_for_channels([channel_group[0]]):
            ray_type = efield.get_parameter(efp.ray_path_type)
            rec_energy_fluences[i_event, i_group, ray_type - 1] = efield.get_parameter(efp.signal_energy_fluence)['{:.0f}-{:.0f}'.format(
                passbands[0][0] / units.MHz,
                passbands[0][1] / units.MHz
            )]
            has_rec_efield[i_event, i_group, ray_type - 1] = True
            rec_energy_fluence = trace_utilities.get_electric_field_energy_fluence(
                efield.get_filtered_trace(passbands[0], 'butter', 10),
                efield.get_times()
            )
            rec_energy_fluence_2 = trace_utilities.get_electric_field_energy_fluence(
                efield.get_filtered_trace(passbands[1], 'butter', 10),
                efield.get_times()
            )
            slope = rec_energy_fluence[1] / rec_energy_fluence_2[1]
            log_s_parameter = np.log10(slope)
            rec_slopes[i_event, i_group, ray_type - 1] = slope
            rec_energy_parameters[i_event, i_group, ray_type - 1] = 1. / (
                    fit_parameters[0] * log_s_parameter**2 + fit_parameters[1] * log_s_parameter + fit_parameters[2]
            )
        for sim_efield in sim_station.get_electric_fields_for_channels([channel_group[0]]):
            ray_type = ray_types.index(sim_efield.get_parameter(efp.ray_path_type))
            energy_fluence = trace_utilities.get_electric_field_energy_fluence(
                sim_efield.get_filtered_trace(passbands[0], 'butter', 10),
                sim_efield.get_times()
            )
            energy_fluence_2 = trace_utilities.get_electric_field_energy_fluence(
                sim_efield.get_filtered_trace(passbands[1], 'butter', 10),
                sim_efield.get_times()
            )
            energy_fluence[0] = np.sum(energy_fluence)
            sim_energy_fluences[i_event, i_group, ray_type] = energy_fluence
            slope = energy_fluence[1] / energy_fluence_2[1]
            log_s_parameter = np.log10(slope)
            sim_energy_parameters[i_event, i_group, ray_type] = 1. / (
                    fit_parameters[0] * log_s_parameter ** 2 + fit_parameters[1] * log_s_parameter + fit_parameters[2]
            )
            sim_slopes[i_event, i_group, ray_type] = slope
            has_sim_efield[i_event, i_group, ray_type] = True
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
                snr = .5 * (np.max(sim_channel_sum.get_trace()) - np.min(sim_channel_sum.get_trace())) / noise_rms
                if max_snr < snr:
                    max_snrs[i_event, i_group] = snr
                    max_snr = snr
fig1 = plt.figure(figsize=(8, 8))
x_points = np.power(10., np.arange(-2, 4, .5))

for i_group, channel_group in enumerate(channel_groups):
    ax1_1 = fig1.add_subplot(len(channel_groups) // 2 + len(channel_groups) % 2, 2, i_group + 1)
    ax1_1.grid()
    has_entry_mask = has_rec_efield[:, i_group] & has_sim_efield[:, i_group]
    if len(channel_group) > 1:
        pol = 0
        ax1_1.set_xlabel(r'$\Phi^E_{sim}$')
        ax1_1.set_ylabel(r'$\Phi^E_{rec}$')
    else:
        pol = 1
        ax1_1.set_xlabel(r'$\Phi^\theta_{sim}$')
        ax1_1.set_ylabel(r'$\Phi^\theta_{rec}$')
    energy_fluence_scatter = ax1_1.scatter(
        sim_energy_fluences[:, i_group, :, pol][has_entry_mask],
        rec_energy_fluences[:, i_group, :, pol][has_entry_mask],
        c=np.tile(max_snrs[:, i_group], (3, 1)).T[has_entry_mask],
        vmin=0,
        vmax=10,
        cmap=cmap
    )
    ax1_1.plot(x_points, x_points, color='k')
    ax1_1.plot(x_points, .5 * x_points, color='k', linestyle=':')
    ax1_1.plot(x_points, 2. * x_points, color='k', linestyle=':')
    plt.colorbar(energy_fluence_scatter, ax=ax1_1)
    ax1_1.set_title('Channels {}'.format(channel_group))
    ax1_1.set_xscale('log')
    ax1_1.set_yscale('log')
    ax1_1.set_xlim([.1, 1.e3])
    ax1_1.set_ylim([.1, 1.e3])
fig1.tight_layout()
fig1.savefig('plots/efield_reco_scatter.png')

fig2 = plt.figure(figsize=(8, 8))
for i_group, channel_group in enumerate(channel_groups):
    ax2_1 = fig2.add_subplot(len(channel_groups) // 2 + len(channel_groups) % 2, 2, i_group + 1)
    ax2_1.grid()
    has_entry_mask = has_rec_efield[:, i_group] & has_sim_efield[:, i_group]
    slope_scatter = ax2_1.scatter(
        sim_slopes[:, i_group][has_entry_mask],
        rec_slopes[:, i_group][has_entry_mask],
        c=np.tile(max_snrs[:, i_group], (3, 1)).T[has_entry_mask],
        vmin=0,
        vmax=10,
        cmap=cmap
    )
    plt.colorbar(slope_scatter, ax=ax2_1)
    ax2_1.set_xscale('log')
    ax2_1.set_yscale('log')
    ax2_1.set_xlim([.1, 1.e3])
    ax2_1.set_ylim([.1, 1.e3])
    plt.plot(x_points, x_points, color='k')
fig2.tight_layout()
fig2.savefig('plots/efield_slope_scatter.png')

fig3 = plt.figure(figsize=(8, 8))
for i_group, channel_group in enumerate(channel_groups):
    ax3_1 = fig3.add_subplot(len(channel_groups) // 2 + len(channel_groups) % 2, 2, i_group + 1)
    ax3_1.grid()
    has_entry_mask = has_rec_efield[:, i_group] & has_sim_efield[:, i_group]
    energy_factor_scatter = ax3_1.scatter(
        sim_energy_parameters[:, i_group][has_entry_mask],
        rec_energy_parameters[:, i_group][has_entry_mask],
        c=np.tile(max_snrs[:, i_group], (3, 1)).T[has_entry_mask],
        vmin=0,
        vmax=10,
        cmap=cmap
    )
    plt.colorbar(energy_factor_scatter, ax=ax3_1)
    ax3_1.set_xscale('log')
    ax3_1.set_yscale('log')
    ax3_1.set_xlim([.01, .5])
    ax3_1.set_ylim([.01, .5])
    plt.plot(x_points, x_points, color='k')
    plt.plot(x_points, .5 * x_points, color='k', linestyle=':')
    plt.plot(x_points, 2. * x_points, color='k', linestyle=':')
fig3.tight_layout()
fig3.savefig('plots/energy_factor_scatter.png')

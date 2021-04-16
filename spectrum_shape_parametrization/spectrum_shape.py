import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from NuRadioReco.utilities import units, bandpass_filter, fft
from NuRadioReco.utilities.trace_utilities import conversion_factor_integrated_signal
import NuRadioMC.utilities.medium
import NuRadioMC.SignalGen.askaryan
import pickle

energies = np.power(10, np.arange(15, 19.1, 1.))
viewing_angles = np.arange(-10, 10, .5) * units.deg
ice = NuRadioMC.utilities.medium.get_ice_model('greenland_simple')
ior = ice.get_index_of_refraction([0, 0, -1000])
cherenkov_angle = np.arccos(1. / ior)
n_samples = 1024
sampling_rate = 10.
model = 'ARZ2019'
shower_types = ['HAD', 'EM']
freqs = np.fft.rfftfreq(n_samples, 1. / sampling_rate)
times = np.arange(n_samples) / sampling_rate
cmap = plt.get_cmap('seismic', 10)
passband_groups = [
    [
        [.13, .3],
        [.3, .5]
    ],
    [
        [.13, .2],
        [.2, .3]
    ]
]
fit_parameters = {}
fontsize = 14


fig2 = plt.figure(figsize=(6, 8))
for i_group, passband_group in enumerate(passband_groups):
    ax2_1 = fig2.add_subplot(len(passband_groups), 1, i_group + 1)
    slope_fluences = np.zeros((len(energies), len(viewing_angles), 2, 2))
    f = []
    s = []
    for i_energy, energy in enumerate(energies):
        for i_type, shower_type in enumerate(shower_types):
            for i_angle, viewing_angle in enumerate(viewing_angles):
                spec = NuRadioMC.SignalGen.askaryan.get_frequency_spectrum(
                    energy=energy,
                    theta=viewing_angle + cherenkov_angle,
                    N=n_samples,
                    dt=1. / sampling_rate,
                    shower_type=shower_type,
                    n_index=ior,
                    model=model,
                    R=1. * units.km,
                    same_shower=False,
                    shift_for_xmax=True
                )
                for i_band, passband in enumerate(passband_group):
                    filtered_trace = spec * bandpass_filter.get_filter_response(freqs, passband, 'butter', 10)
                    slope_fluences[i_energy, i_angle, i_type, i_band] = np.sum(fft.freq2time(filtered_trace, sampling_rate)**2) * conversion_factor_integrated_signal / sampling_rate
                if i_type == 0:
                    f.append(np.sqrt(slope_fluences[i_energy, i_angle, 0, 0]) / (energy / units.EeV))
                    s.append(slope_fluences[i_energy, i_angle, 0, 0] / slope_fluences[i_energy, i_angle, 0, 1])
        fluence_scatter = ax2_1.scatter(
            (slope_fluences[i_energy, :, 0, 0] / slope_fluences[i_energy, :, 0, 1]),
            np.sqrt(slope_fluences[i_energy, :, 0, 0]) / (energy / units.EeV),
            c=viewing_angles / units.deg,
            vmin=-10,
            vmax=10,
            cmap=cmap
        )
        ax2_1.scatter(
            (slope_fluences[i_energy, :, 1, 0] / slope_fluences[i_energy, :, 1, 1]),
            np.sqrt(slope_fluences[i_energy, :, 1, 0]) / (energy / units.EeV),
            c=viewing_angles / units.deg,
            vmin=-10,
            vmax=10,
            cmap=cmap,
            marker='x',
            alpha=1.
        )
    fit = np.polyfit(np.log10(s), f, 2)
    x_points = np.power(10., np.arange(-1, 2, .1))
    y_points = fit[0] * np.log10(x_points)**2 + fit[1] * np.log10(x_points) + fit[2]
    fit_parameters[
        '{:.0f}-{:.0f}/{:.0f}-{:.0f}'.format(
            passband_group[0][0] / units.MHz,
            passband_group[0][1] / units.MHz,
            passband_group[1][0] / units.MHz,
            passband_group[1][1] / units.MHz
        )
    ] = fit
    plt.colorbar(fluence_scatter, ax=ax2_1).set_label(r'$\varphi - \varphi_{Cherenkov} [^\circ]$', fontsize=fontsize)
    ax2_1.grid()
    legend_elements = [
        Line2D([0], [0], markerfacecolor='k', color='none', marker='o', label='hadronic'),
        Line2D([0], [0], markerfacecolor='k', color='none', marker='x', label='electro-magnetic')
    ]
    plt.legend(handles=legend_elements, fontsize=fontsize)
    ax2_1.plot(x_points, y_points, color='k')
    ax2_1.set_xscale('log')
    ax2_1.set_facecolor('silver')
    ax2_1.set_xlim([1.e-1, 1.e2])
    ax2_1.set_ylim([0, 50])
    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    ax2_1.set_xlabel(r'$\Phi^E_{%.0f-%.0f} / \Phi^E_{%.0f-%.0f}$' % (passband_group[0][0] / units.MHz, passband_group[0][1] / units.MHz, passband_group[1][0] / units.MHz, passband_group[1][1] / units.MHz), fontsize=fontsize)
    ax2_1.set_ylabel(r'$\sqrt{\Phi^E_{%.0f-%.0f}} / (E/EeV)$' % (passband_group[0][0] / units.MHz, passband_group[0][1] / units.MHz), fontsize=fontsize)
fig2.tight_layout()
fig2.savefig('plots/parametrizations.png')

pickle.dump(fit_parameters, open('fit_parameters.p', 'wb'))

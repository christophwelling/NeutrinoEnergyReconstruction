import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines
from NuRadioReco.utilities import units, fft
import NuRadioMC.SignalGen.askaryan
import NuRadioMC.utilities.medium

energy = 1.e18 * units.eV
ice = NuRadioMC.utilities.medium.get_ice_model('greenland_simple')
index_of_refraction = ice.get_index_of_refraction([0, 0, -1000])
cherenkov_angle = np.arccos(1. / index_of_refraction)
n_samples = 1024
sampling_rate = 5. * units.GHz
viewing_angles = np.arange(0, 4, 1.) * units.deg
distance = 1. * units.km
freqs = np.fft.rfftfreq(n_samples, 1. / sampling_rate)
inelasticity = .25

fig1 = plt.figure(figsize=(6, 3))
ax1_1 = fig1.add_subplot(121)
ax1_1.grid()
ax1_2 = fig1.add_subplot(122, sharey=ax1_1)
ax1_2.grid()

for i_angle, viewing_angle in enumerate(viewing_angles):
    efield_spectrum_had = NuRadioMC.SignalGen.askaryan.get_frequency_spectrum(
        energy=energy,
        theta=cherenkov_angle + viewing_angle,
        N=n_samples,
        dt=1. / sampling_rate,
        shower_type='HAD',
        n_index=index_of_refraction,
        R=distance,
        model='ARZ2020',
        same_shower=True
    )
    efield_spectrum_em = NuRadioMC.SignalGen.askaryan.get_frequency_spectrum(
        energy=energy,
        theta=cherenkov_angle + viewing_angle,
        N=n_samples,
        dt=1. / sampling_rate,
        shower_type='EM',
        n_index=index_of_refraction,
        R=distance,
        model='ARZ2020',
        same_shower = True
    )
    ax1_1.plot(
        freqs / units.MHz,
        np.abs(inelasticity * efield_spectrum_had) / (units.microvolt / units.m / units.MHz),
        c='C{}'.format(i_angle),
        label=r'$\varphi - \varphi_{Cherenkov}=%.0f ^\circ$' % (viewing_angle / units.deg),
        linewidth=2
    )
    ax1_1.plot(
        freqs / units.MHz,
        np.abs((1 - inelasticity) * efield_spectrum_em) / (units.microvolt / units.m / units.MHz),
        c='C{}'.format(i_angle),
        linestyle=':',
        linewidth=2
    )
    ax1_2.plot(
        freqs / units.MHz,
        np.abs(inelasticity * efield_spectrum_had + (1 - inelasticity) * efield_spectrum_em) / (units.microvolt / units.m / units.MHz),
        c='C{}'.format(i_angle),
        linewidth=2
    )


ax1_1.set_xlim([0, 990])
ax1_1.set_xlabel('f [MHz]')
ax1_1.set_ylabel(r'E [$\mu$V/m/MHz]')
ax1_1.set_ylim([0, None])
ax1_2.set_xlim([0, 990])
ax1_2.set_xlabel('f [MHz]')
for tick in ax1_2.get_yticklabels():
    tick.set_visible(False)

handles, labels = ax1_1.get_legend_handles_labels()
fig1.legend(handles, labels, ncol=2, bbox_to_anchor=(.725, 1.01))

fig1.tight_layout()
plt.subplots_adjust(wspace=.01)
box1_1 = ax1_1.get_position()
ax1_1.set_position([
    box1_1.x0,
    box1_1.y0,
    box1_1.width,
    .8 * box1_1.height
])
box1_2 = ax1_2.get_position()
ax1_2.set_position([
    box1_2.x0,
    box1_2.y0,
    box1_2.width,
    .8 * box1_2.height
])
fig1.savefig('plots/neutrino_efield_spectra.png')


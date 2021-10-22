import numpy as np
import matplotlib.pyplot as plt
import NuRadioReco.modules.io.coreas.readCoREASShower
import NuRadioReco.detector.generic_detector
import radiotools.coordinatesystems
import radiotools.helper
from NuRadioReco.framework.parameters import showerParameters as shp
from NuRadioReco.utilities import units, fft
import glob

generic_detector = NuRadioReco.detector.generic_detector.GenericDetector(
    json_filename='RNO_single_station.json',
    default_station=0
)

coreas_reader = NuRadioReco.modules.io.coreas.readCoREASShower.readCoREASShower()
coreas_reader.begin(
    glob.glob('*.hdf5'),
    det=generic_detector
)

plot_distances = np.arange(0, 510, 100)

for i_event, (event, det) in enumerate(coreas_reader.run()):
    station_ids = np.array(event.get_station_ids())
    station_positions = np.zeros((len(station_ids), 3))
    shower = list(event.get_sim_showers())[0]
    cs_trafo = radiotools.coordinatesystems.cstrafo(
        shower.get_parameter(shp.zenith),
        shower.get_parameter(shp.azimuth),
        shower.get_parameter(shp.magnetic_field_vector)
        # radiotools.helper.get_magnetic_field_vector('mooresbay')
    )
    print(shower.get_parameter(shp.zenith) / units.deg)
    for i_station, station_id in enumerate(event.get_station_ids()):
        station = event.get_station(station_id)
        station_position = det.get_absolute_position(station_id)
        station_positions[i_station] = station_position
    station_positions_vxB = cs_trafo.transform_to_vxB_vxvxB(station_positions, core=[0, 0, station_positions[0, 2]])
    station_distances = np.sqrt(station_positions_vxB[:, 0]**2 + station_positions_vxB[:, 1]**2)
    station_position_mask = (station_positions_vxB[:, 0] > 0)
    station_branch_mask = (np.abs(np.arctan(station_positions_vxB[:, 1] / station_positions_vxB[:, 0])) < 30 * units.deg)
    fig1 = plt.figure()
    ax1_1 = fig1.add_subplot(111)
    ax1_1.grid()
    ax1_1.scatter(
        station_positions_vxB[:, 0][station_position_mask & station_branch_mask],
        station_positions_vxB[:, 1][station_position_mask & station_branch_mask],
        c=station_distances[station_position_mask & station_branch_mask]
    )
    ax1_1.set_aspect('equal')
    fig1.tight_layout()
    fig1.savefig('plots/footprint_{}.png'.format(event.get_id()))

    fig2 = plt.figure(figsize=(6, 3))
    ax2_1 = fig2.add_subplot(111)
    ax2_1.grid()
    # ax2_1.set_yscale('log')
    ax2_1.set_xlim([0, 500])
    ax2_1.set_title(r'$\lg_{10}(E)=%.1f, \;\theta=%.1f$' % (np.log10(shower.get_parameter(shp.energy)), shower.get_parameter(shp.zenith) / units.deg))
    ax2_1.set_ylim([0, 20])
    ax2_1.set_xlabel('f [MHz]')
    ax2_1.set_ylabel(r'E [$\mu V/m/MHz$]')
    d_xmax = shower.get_parameter(shp.distance_shower_maximum_geometric)
    for i_distance, plot_distance in enumerate(plot_distances):
        station_id = station_ids[station_position_mask & station_branch_mask][np.argmin(np.abs(plot_distance - station_distances[station_position_mask & station_branch_mask]))]
        sim_station = event.get_station(station_id).get_sim_station()
        station_distance = station_distances[station_id == station_ids][0]
        viewing_angle = np.arctan(station_distance / d_xmax)
        for efield in sim_station.get_electric_fields():
            efield_trace_vxB = cs_trafo.transform_to_vxB_vxvxB(efield.get_trace())
            efield_spectrum_vxB = fft.time2freq(efield_trace_vxB, efield.get_sampling_rate())
            ax2_1.plot(
                efield.get_frequencies() / units.MHz,
                np.sqrt(np.abs(efield_spectrum_vxB[0])**2 + np.abs(efield_spectrum_vxB[1]**2)) / (units.microvolt / units.m / units.MHz),
                label=r'd = %.0f m, $\varphi\approx$%.1f' % (station_distance, viewing_angle / units.deg),
                color='C{}'.format(i_distance),
                linewidth=2
            )
    ax2_1.legend(ncol=2)
    fig2.tight_layout()
    fig2.savefig('plots/efield_spectra_{}.png'.format(i_event))

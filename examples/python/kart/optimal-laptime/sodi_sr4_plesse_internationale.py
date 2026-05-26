import inspect
import os
import sys

import numpy as np
from matplotlib.animation import FuncAnimation, FFMpegWriter
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.patches import Circle, Rectangle

sys.path.append(
    os.path.dirname(
        os.path.dirname(
            os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
        )
    )
)

import fastest_lap

DRY_VEHICLE_NAME = "sodi_sr4_390cc_dry"
WET_VEHICLE_NAME = "sodi_sr4_390cc_wet"
TRACK_NAME = "plesse_internationale_karting"
ARTIFACTS_DIR = os.path.join(
    os.path.dirname(__file__), "artifacts", "sodi_sr4_plesse_internationale"
)
WET_ARTIFACTS_DIR = os.path.join(
    os.path.dirname(__file__), "artifacts", "sodi_sr4_plesse_internationale_wet_line"
)
COMPARISON_ARTIFACTS_DIR = os.path.join(
    os.path.dirname(__file__), "artifacts", "sodi_sr4_plesse_internationale_comparison"
)
WET_BASE_GRIP_MULTIPLIER = 0.74
WET_DRY_LINE_PENALTY = 0.28
WET_DRY_LINE_WIDTH_M = 0.8
ACCENT = "#ffb000"
ACCENT_RED = "#ff5f57"
ACCENT_GREEN = "#58c472"
BRAKE_LINE_CMAP = LinearSegmentedColormap.from_list(
    "brake_gradient", ["#ffd7d7", "#ff0000"]
)
THROTTLE_LINE_CMAP = LinearSegmentedColormap.from_list(
    "throttle_gradient", ["#d7ffd7", "#00d000"]
)
PERCENT_NORM = Normalize(vmin=0.0, vmax=1.0)
BRAKE_DISPLAY_THRESHOLD = 1.0e-4


def build_racing_line_segments(x, y, pedal_signal):
    points = np.column_stack((np.asarray(x), np.asarray(y)))
    if len(points) < 2:
        return np.empty((0, 2, 2)), np.empty((0, 4))

    pedal_signal = np.clip(np.asarray(pedal_signal), -1.0, 1.0)
    throttle = np.clip(pedal_signal, 0.0, 1.0)
    brake = np.clip(-pedal_signal, 0.0, 1.0)
    segments = np.stack((points[:-1], points[1:]), axis=1)
    segment_throttle = 0.5 * (throttle[:-1] + throttle[1:])
    segment_brake = 0.5 * (brake[:-1] + brake[1:])
    segment_colors = THROTTLE_LINE_CMAP(PERCENT_NORM(segment_throttle))

    braking_mask = segment_brake > BRAKE_DISPLAY_THRESHOLD
    segment_colors[braking_mask] = BRAKE_LINE_CMAP(
        PERCENT_NORM(segment_brake[braking_mask])
    )

    return segments, segment_colors


def interpolate_realtime_samples(run, fps):
    lap_time = np.array(run["time"])
    total_time = float(lap_time[-1])
    time_samples = np.arange(0.0, total_time, 1.0 / fps)
    if time_samples[-1] < total_time:
        time_samples = np.append(time_samples, total_time)

    yaw = np.unwrap(np.array(run["chassis.attitude.yaw"]))
    throttle_signal = np.array(run["rear-axle.throttle"])

    samples = {
        "time": time_samples,
        "x": np.interp(time_samples, lap_time, np.array(run["chassis.position.x"])),
        "y": np.interp(time_samples, lap_time, np.array(run["chassis.position.y"])),
        "speed_kmh": np.interp(
            time_samples, lap_time, np.array(run["chassis.velocity.x"]) * 3.6
        ),
        "steering_deg": np.interp(
            time_samples,
            lap_time,
            np.array(run["front-axle.steering-angle"]) * 180.0 / np.pi,
        ),
        "pedal_signal": np.interp(time_samples, lap_time, throttle_signal),
        "yaw": np.interp(time_samples, lap_time, yaw),
        "arclength": np.interp(time_samples, lap_time, np.array(run["road.arclength"])),
    }
    samples["throttle"] = np.clip(samples["pedal_signal"], 0.0, 1.0)
    samples["brake"] = np.clip(-samples["pedal_signal"], 0.0, 1.0)

    return samples


def save_realtime_video(
    run,
    track_coordinates,
    artifacts_dir=ARTIFACTS_DIR,
    title="Sodi SR4 390cc at Plesse Internationale",
    video_filename="plesse-internationale_sodi-sr4.mp4",
):
    if fastest_lap.plt is None:
        raise RuntimeError("matplotlib is not available; cannot generate video")

    plt = fastest_lap.plt
    fps = 20
    samples = interpolate_realtime_samples(run, fps)

    x_center, y_center, x_left, y_left, x_right, y_right, _ = track_coordinates
    speed_trace_s = np.array(run["road.arclength"])
    speed_trace_kmh = np.array(run["chassis.velocity.x"]) * 3.6
    throttle_trace = np.clip(np.array(run["rear-axle.throttle"]), 0.0, 1.0)
    brake_trace = np.clip(-np.array(run["rear-axle.throttle"]), 0.0, 1.0)
    lap_segments, lap_segment_colors = build_racing_line_segments(
        samples["x"], samples["y"], samples["pedal_signal"]
    )
    steering_trace_deg = np.array(run["front-axle.steering-angle"]) * 180.0 / np.pi
    steering_display_ratio = float(
        np.clip(135.0 / max(1.0, np.max(np.abs(steering_trace_deg))), 12.0, 45.0)
    )

    bg = "#0b1220"
    card = "#131c2e"
    card_2 = "#18233a"
    grid_color = "#51607a"
    text_main = "#edf2ff"
    text_muted = "#a8b3c9"
    accent = ACCENT
    accent_red = ACCENT_RED
    accent_green = ACCENT_GREEN
    track_color = "#90a4c3"

    def style_axis(ax, title=None):
        ax.set_facecolor(card)
        for spine in ax.spines.values():
            spine.set_color("#31405a")
            spine.set_linewidth(1.0)
        ax.tick_params(colors=text_muted, labelsize=11)
        ax.xaxis.label.set_color(text_main)
        ax.yaxis.label.set_color(text_main)
        if title is not None:
            ax.set_title(title, color=text_main, fontsize=16, fontweight="bold", pad=10)

    fig = plt.figure(figsize=(16, 9), dpi=160)
    fig.patch.set_facecolor(bg)
    grid = fig.add_gridspec(
        3,
        5,
        height_ratios=[4.8, 1.45, 1.45],
        width_ratios=[1.95, 1.95, 1.95, 1.95, 0.85],
        hspace=0.26,
        wspace=0.08,
    )
    fig.subplots_adjust(left=0.05, right=0.97, top=0.95, bottom=0.05)
    track_ax = fig.add_subplot(grid[0, :4])
    telemetry_ax = fig.add_subplot(grid[0, 4])
    speed_ax = fig.add_subplot(grid[1, :])
    pedals_ax = fig.add_subplot(grid[2, :])

    style_axis(track_ax, title)
    track_ax.set_aspect("equal", adjustable="datalim")
    track_ax.set_anchor("W")
    track_ax.grid(False)
    track_ax.plot(
        x_center,
        y_center,
        linewidth=0.7,
        color=track_color,
        linestyle=(0, (20, 4)),
        alpha=0.75,
    )
    track_ax.plot(x_left, y_left, linewidth=1.2, color=track_color, alpha=0.9)
    track_ax.plot(x_right, y_right, linewidth=1.2, color=track_color, alpha=0.9)
    lap_line = LineCollection(
        np.empty((0, 2, 2)),
        linewidth=1.35,
        zorder=4,
    )
    track_ax.add_collection(lap_line)
    (car_marker,) = track_ax.plot(
        [],
        [],
        marker="o",
        markersize=7.0,
        color=accent_red,
        markeredgecolor="white",
        markeredgewidth=0.6,
    )
    car_heading = track_ax.quiver(
        [],
        [],
        [],
        [],
        angles="xy",
        scale_units="xy",
        scale=1.0,
        color=accent_red,
        width=0.004,
    )
    track_hud = track_ax.text(
        0.03,
        0.92,
        "",
        transform=track_ax.transAxes,
        va="top",
        ha="left",
        color=text_main,
        fontsize=13,
        linespacing=1.35,
        bbox={
            "facecolor": card_2,
            "alpha": 0.92,
            "edgecolor": "#2a3953",
            "boxstyle": "round,pad=0.45",
        },
    )
    track_ax.invert_yaxis()
    x_all = np.concatenate(
        (np.asarray(x_left), np.asarray(x_right), np.asarray(x_center))
    )
    y_all = np.concatenate(
        (np.asarray(y_left), np.asarray(y_right), np.asarray(y_center))
    )
    x_pad = 0.04 * float(np.max(x_all) - np.min(x_all))
    y_pad = 0.04 * float(np.max(y_all) - np.min(y_all))
    track_ax.set_xlim(float(np.min(x_all) - x_pad), float(np.max(x_all) + x_pad))
    track_ax.set_ylim(float(np.max(y_all) + y_pad), float(np.min(y_all) - y_pad))
    track_ax.set_xticks([])
    track_ax.set_yticks([])

    telemetry_ax.set_facecolor(card)
    telemetry_ax.set_xlim(0.0, 1.0)
    telemetry_ax.set_ylim(0.0, 1.0)
    telemetry_ax.set_aspect("equal", adjustable="box")
    telemetry_ax.axis("off")

    wheel_center = np.array([0.5, 0.78])
    wheel_radius = 0.18
    telemetry_ax.add_patch(
        Circle(
            wheel_center,
            wheel_radius,
            fill=False,
            linewidth=3.5,
            edgecolor=text_main,
            alpha=0.95,
        )
    )
    telemetry_ax.add_patch(
        Circle(
            wheel_center,
            wheel_radius * 0.22,
            fill=False,
            linewidth=2.1,
            edgecolor=text_muted,
            alpha=0.95,
        )
    )
    (steering_horizontal,) = telemetry_ax.plot(
        [], [], color=text_main, linewidth=3.2, solid_capstyle="round"
    )
    (steering_vertical,) = telemetry_ax.plot(
        [], [], color=text_main, linewidth=3.2, solid_capstyle="round"
    )

    pedal_base_y = 0.015
    pedal_height = 0.50
    pedal_width = 0.14
    pedal_left_x = 0.25
    pedal_right_x = 0.61
    for x0 in (pedal_left_x, pedal_right_x):
        telemetry_ax.add_patch(
            Rectangle(
                (x0, pedal_base_y),
                pedal_width,
                pedal_height,
                fill=False,
                linewidth=1.4,
                edgecolor="#4a5b79",
            )
        )
    brake_fill = Rectangle(
        (pedal_left_x, pedal_base_y), pedal_width, 0.0, color=accent_red, alpha=0.9
    )
    throttle_fill = Rectangle(
        (pedal_right_x, pedal_base_y), pedal_width, 0.0, color=accent_green, alpha=0.9
    )
    telemetry_ax.add_patch(brake_fill)
    telemetry_ax.add_patch(throttle_fill)

    style_axis(speed_ax, "Speed At Distance")
    speed_ax.plot(
        speed_trace_s, speed_trace_kmh, color="#6e7f9f", linewidth=0.85, alpha=0.42
    )
    (speed_progress,) = speed_ax.plot([], [], color=accent, linewidth=1.1)
    speed_cursor = speed_ax.axvline(0.0, color=accent_red, linewidth=1.0, alpha=0.95)
    (speed_marker,) = speed_ax.plot(
        [],
        [],
        marker="o",
        color=accent_red,
        markeredgecolor="white",
        markeredgewidth=0.5,
        markersize=5.0,
    )
    speed_ax.set_title("Speed", color=text_main, fontsize=16, fontweight="bold", pad=8)
    speed_ax.set_ylabel("Speed [km/h]")
    speed_ax.set_xlim(0.0, float(speed_trace_s[-1]))
    speed_ax.set_ylim(0.0, max(110.0, float(np.max(speed_trace_kmh) * 1.08)))
    speed_ax.grid(True, alpha=0.24, color=grid_color)
    speed_ax.tick_params(axis="x", labelbottom=False)

    style_axis(pedals_ax, "Pedals At Distance")
    pedals_ax.plot(
        speed_trace_s, brake_trace * 100.0, color=accent_red, linewidth=0.7, alpha=0.32
    )
    pedals_ax.plot(
        speed_trace_s,
        throttle_trace * 100.0,
        color=accent_green,
        linewidth=0.7,
        alpha=0.32,
    )
    (brake_progress,) = pedals_ax.plot([], [], color=accent_red, linewidth=0.9)
    (throttle_progress,) = pedals_ax.plot([], [], color=accent_green, linewidth=0.9)
    pedals_cursor = pedals_ax.axvline(0.0, color=accent_red, linewidth=1.0, alpha=0.95)
    (brake_marker,) = pedals_ax.plot(
        [],
        [],
        marker="o",
        color=accent_red,
        markeredgecolor="white",
        markeredgewidth=0.5,
        markersize=5.0,
    )
    (throttle_marker,) = pedals_ax.plot(
        [],
        [],
        marker="o",
        color=accent_green,
        markeredgecolor="white",
        markeredgewidth=0.5,
        markersize=5.0,
    )
    pedals_ax.set_title(
        "Pedals", color=text_main, fontsize=16, fontweight="bold", pad=8
    )
    pedals_ax.set_xlabel("Distance [m]")
    pedals_ax.set_ylabel("Pedal [%]")
    pedals_ax.set_xlim(0.0, float(speed_trace_s[-1]))
    pedals_ax.set_ylim(-3.0, 103.0)
    pedals_ax.grid(True, alpha=0.24, color=grid_color)

    heading_length = (
        max(
            float(np.max(x_left) - np.min(x_right)),
            float(np.max(y_left) - np.min(y_right)),
        )
        * 0.03
    )

    def update(frame_number):
        nonlocal car_heading

        current_time = samples["time"][frame_number]
        current_x = samples["x"][frame_number]
        current_y = samples["y"][frame_number]
        current_speed = samples["speed_kmh"][frame_number]
        current_steer = samples["steering_deg"][frame_number]
        current_throttle = samples["throttle"][frame_number]
        current_brake = samples["brake"][frame_number]
        current_s = samples["arclength"][frame_number]
        current_yaw = samples["yaw"][frame_number]

        if frame_number == 0:
            lap_line.set_segments([])
        else:
            lap_line.set_segments(lap_segments[:frame_number])
            lap_line.set_color(lap_segment_colors[:frame_number])
        car_marker.set_data([current_x], [current_y])

        car_heading.remove()
        car_heading = track_ax.quiver(
            [current_x],
            [current_y],
            [heading_length * np.cos(current_yaw)],
            [heading_length * np.sin(current_yaw)],
            angles="xy",
            scale_units="xy",
            scale=1.0,
            color="red",
            width=0.004,
        )

        track_hud.set_text(
            f"Lap time: {current_time:6.1f} s\n"
            f"Distance: {current_s:6.0f} m\n"
            f"Speed:    {current_speed:6.1f} km/h"
        )

        brake_fill.set_height(current_brake * pedal_height)
        throttle_fill.set_height(current_throttle * pedal_height)
        wheel_rotation = -np.deg2rad(current_steer * steering_display_ratio)
        horizontal_dx = wheel_radius * 0.86 * np.cos(wheel_rotation)
        horizontal_dy = wheel_radius * 0.86 * np.sin(wheel_rotation)
        steering_horizontal.set_data(
            [wheel_center[0] - horizontal_dx, wheel_center[0] + horizontal_dx],
            [wheel_center[1] - horizontal_dy, wheel_center[1] + horizontal_dy],
        )
        vertical_angle = np.deg2rad(90.0) + wheel_rotation
        steering_vertical.set_data(
            [
                wheel_center[0],
                wheel_center[0] + wheel_radius * 0.92 * np.cos(vertical_angle),
            ],
            [
                wheel_center[1],
                wheel_center[1] + wheel_radius * 0.92 * np.sin(vertical_angle),
            ],
        )

        trace_idx = np.searchsorted(speed_trace_s, current_s, side="right")
        speed_progress.set_data(speed_trace_s[:trace_idx], speed_trace_kmh[:trace_idx])
        speed_cursor.set_xdata([current_s, current_s])
        speed_marker.set_data([current_s], [current_speed])

        brake_progress.set_data(
            speed_trace_s[:trace_idx], brake_trace[:trace_idx] * 100.0
        )
        throttle_progress.set_data(
            speed_trace_s[:trace_idx], throttle_trace[:trace_idx] * 100.0
        )
        pedals_cursor.set_xdata([current_s, current_s])
        brake_marker.set_data([current_s], [current_brake * 100.0])
        throttle_marker.set_data([current_s], [current_throttle * 100.0])

        return (
            lap_line,
            car_marker,
            car_heading,
            track_hud,
            steering_horizontal,
            steering_vertical,
            brake_fill,
            throttle_fill,
            speed_progress,
            speed_cursor,
            speed_marker,
            brake_progress,
            throttle_progress,
            pedals_cursor,
            brake_marker,
            throttle_marker,
        )

    animation = FuncAnimation(
        fig,
        update,
        frames=len(samples["time"]),
        interval=1000.0 / fps,
        blit=False,
    )

    video_path = os.path.join(artifacts_dir, video_filename)
    writer = FFMpegWriter(
        fps=fps, codec="libx264", bitrate=6000, extra_args=["-pix_fmt", "yuv420p"]
    )
    animation.save(video_path, writer=writer)
    plt.close(fig)

    return video_path


def save_visualizations(
    run,
    artifacts_dir=ARTIFACTS_DIR,
    title="Sodi SR4 390cc at Plesse Internationale",
    video_filename="plesse-internationale_sodi-sr4.mp4",
    make_video=True,
):
    if fastest_lap.plt is None:
        raise RuntimeError(
            "matplotlib is not available; cannot generate visualizations"
        )

    plt = fastest_lap.plt
    os.makedirs(artifacts_dir, exist_ok=True)

    s = np.array(run["road.arclength"])
    x = np.array(run["chassis.position.x"])
    y = np.array(run["chassis.position.y"])
    speed_kmh = np.array(run["chassis.velocity.x"]) * 3.6
    steering_deg = np.array(run["front-axle.steering-angle"]) * 180.0 / np.pi
    pedal_signal = np.array(run["rear-axle.throttle"])
    throttle_trace = np.clip(pedal_signal, 0.0, 1.0)
    brake_trace = np.clip(-pedal_signal, 0.0, 1.0)
    track_segments, track_segment_colors = build_racing_line_segments(
        x, y, pedal_signal
    )
    lap_time = np.array(run["time"])
    x_center, y_center, x_left, y_left, x_right, y_right, _ = (
        fastest_lap.track_coordinates(TRACK_NAME)
    )

    bg = "#ffffff"
    card = "#ffffff"
    grid_color = "#d0d7e2"
    text_main = "#18233a"
    text_muted = "#51607a"
    track_color = "#90a4c3"

    def style_axis(ax, title=None):
        ax.set_facecolor(card)
        for spine in ax.spines.values():
            spine.set_color("#d0d7e2")
            spine.set_linewidth(1.0)
        ax.tick_params(colors=text_muted, labelsize=11)
        ax.xaxis.label.set_color(text_main)
        ax.yaxis.label.set_color(text_main)
        if title is not None:
            ax.set_title(title, color=text_main, fontsize=16, fontweight="bold", pad=10)

    track_fig = plt.figure(figsize=(16, 9), dpi=160)
    track_fig.patch.set_facecolor(bg)
    grid = track_fig.add_gridspec(2, 1, height_ratios=[3.6, 1.15], hspace=0.18)
    track_fig.subplots_adjust(left=0.045, right=0.98, top=0.95, bottom=0.08)
    track_ax = track_fig.add_subplot(grid[0, 0])
    pedals_ax = track_fig.add_subplot(grid[1, 0])

    style_axis(track_ax, title)
    track_ax.set_aspect("equal", adjustable="datalim")
    track_ax.set_anchor("C")
    track_ax.grid(False)
    track_ax.plot(
        x_center,
        y_center,
        linewidth=0.7,
        color=track_color,
        linestyle=(0, (20, 4)),
        alpha=0.75,
    )
    track_ax.plot(x_left, y_left, linewidth=1.2, color=track_color, alpha=0.9)
    track_ax.plot(x_right, y_right, linewidth=1.2, color=track_color, alpha=0.9)
    racing_line = LineCollection(
        track_segments,
        colors=track_segment_colors,
        linewidth=1.35,
        zorder=4,
    )
    track_ax.add_collection(racing_line)
    track_ax.invert_yaxis()
    x_all = np.concatenate((np.asarray(x_left), np.asarray(x_right), np.asarray(x_center)))
    y_all = np.concatenate((np.asarray(y_left), np.asarray(y_right), np.asarray(y_center)))
    x_pad = 0.04 * float(np.max(x_all) - np.min(x_all))
    y_pad = 0.04 * float(np.max(y_all) - np.min(y_all))
    track_ax.set_xlim(float(np.min(x_all) - x_pad), float(np.max(x_all) + x_pad))
    track_ax.set_ylim(float(np.max(y_all) + y_pad), float(np.min(y_all) - y_pad))
    track_ax.set_xticks([])
    track_ax.set_yticks([])

    style_axis(pedals_ax, "Pedals At Distance")
    pedals_ax.plot(
        s, brake_trace * 100.0, color=ACCENT_RED, linewidth=0.7, alpha=0.95
    )
    pedals_ax.plot(
        s, throttle_trace * 100.0, color=ACCENT_GREEN, linewidth=0.7, alpha=0.95
    )
    pedals_ax.set_xlabel("Distance [m]")
    pedals_ax.set_ylabel("Pedal [%]")
    pedals_ax.set_xlim(0.0, float(s[-1]))
    pedals_ax.set_ylim(-3.0, 103.0)
    pedals_ax.grid(True, alpha=0.24, color=grid_color)

    track_fig.savefig(
        os.path.join(artifacts_dir, "track_map.png"),
        dpi=160,
        facecolor=track_fig.get_facecolor(),
    )
    plt.close(track_fig)

    speed_fig = plt.figure(figsize=(14, 4))
    speed_ax = speed_fig.add_subplot(111)
    speed_ax.plot(s, speed_kmh, color="orange", linewidth=2)
    speed_ax.set_title("Speed Trace")
    speed_ax.set_xlabel("Arclength [m]")
    speed_ax.set_ylabel("Speed [km/h]")
    speed_ax.grid(True, alpha=0.3)
    speed_fig.tight_layout()
    speed_fig.savefig(
        os.path.join(artifacts_dir, "speed_trace.png"), dpi=160, bbox_inches="tight"
    )
    plt.close(speed_fig)

    control_fig = plt.figure(figsize=(14, 6))
    steering_ax = control_fig.add_subplot(211)
    steering_ax.plot(s, steering_deg, color="tab:blue", linewidth=1.8)
    steering_ax.set_title("Steering")
    steering_ax.set_ylabel("Angle [deg]")
    steering_ax.grid(True, alpha=0.3)

    throttle_ax = control_fig.add_subplot(212)
    throttle_ax.plot(s, pedal_signal, color="tab:green", linewidth=1.8)
    throttle_ax.set_title("Throttle")
    throttle_ax.set_xlabel("Arclength [m]")
    throttle_ax.set_ylabel("Throttle [-]")
    throttle_ax.grid(True, alpha=0.3)
    control_fig.tight_layout()
    control_fig.savefig(
        os.path.join(artifacts_dir, "controls.png"), dpi=160, bbox_inches="tight"
    )
    plt.close(control_fig)

    lap_time_fig = plt.figure(figsize=(14, 4))
    lap_time_ax = lap_time_fig.add_subplot(111)
    lap_time_ax.plot(s, lap_time, color="tab:red", linewidth=2)
    lap_time_ax.set_title("Accumulated Lap Time")
    lap_time_ax.set_xlabel("Arclength [m]")
    lap_time_ax.set_ylabel("Time [s]")
    lap_time_ax.grid(True, alpha=0.3)
    lap_time_fig.tight_layout()
    lap_time_fig.savefig(
        os.path.join(artifacts_dir, "lap_time_trace.png"), dpi=160, bbox_inches="tight"
    )
    plt.close(lap_time_fig)

    summary_path = os.path.join(artifacts_dir, "summary.txt")
    with open(summary_path, "w", encoding="ascii") as summary_file:
        summary_file.write(f"Laptime: {run['laptime']:.3f} s\n")
        summary_file.write(f"Max speed: {speed_kmh.max():.2f} km/h\n")
        summary_file.write(f"Track points: {len(s)}\n")

    video_path = None
    if make_video:
        video_path = save_realtime_video(
            run,
            (x_center, y_center, x_left, y_left, x_right, y_right, _),
            artifacts_dir,
            title,
            video_filename,
        )

    return artifacts_dir, video_path


def save_comparison(dry_run, wet_run):
    if fastest_lap.plt is None:
        raise RuntimeError("matplotlib is not available; cannot generate comparison")

    plt = fastest_lap.plt
    os.makedirs(COMPARISON_ARTIFACTS_DIR, exist_ok=True)

    dry_s = np.array(dry_run["road.arclength"])
    wet_s = np.array(wet_run["road.arclength"])
    dry_speed_kmh = np.array(dry_run["chassis.velocity.x"]) * 3.6
    wet_speed_kmh = np.array(wet_run["chassis.velocity.x"]) * 3.6
    dry_n = np.array(dry_run["road.lateral-displacement"])
    wet_n = np.array(wet_run["road.lateral-displacement"])
    wet_grip = np.array(wet_run["rear-axle.left-tire.grip-multiplier"])
    wet_line_shift = wet_n - np.interp(wet_s, dry_s, dry_n)

    fig = plt.figure(figsize=(14, 10), dpi=150)
    speed_ax = fig.add_subplot(311)
    line_ax = fig.add_subplot(312)
    grip_ax = fig.add_subplot(313)

    speed_ax.plot(dry_s, dry_speed_kmh, label="Dry", color=ACCENT, linewidth=1.4)
    speed_ax.plot(wet_s, wet_speed_kmh, label="Wet line", color="#2477ff", linewidth=1.4)
    speed_ax.set_title("Dry vs Wet-Line Simulation")
    speed_ax.set_ylabel("Speed [km/h]")
    speed_ax.grid(True, alpha=0.3)
    speed_ax.legend()

    line_ax.plot(dry_s, dry_n, label="Dry reference line", color=ACCENT, linewidth=1.2)
    line_ax.plot(wet_s, wet_n, label="Wet optimized line", color="#2477ff", linewidth=1.2)
    line_ax.set_ylabel("Lateral displacement [m]")
    line_ax.grid(True, alpha=0.3)
    line_ax.legend()

    grip_ax.plot(wet_s, wet_grip, color="#2477ff", linewidth=1.2)
    grip_ax.set_xlabel("Arclength [m]")
    grip_ax.set_ylabel("Wet grip multiplier [-]")
    grip_ax.grid(True, alpha=0.3)

    fig.tight_layout()
    comparison_plot = os.path.join(COMPARISON_ARTIFACTS_DIR, "dry_vs_wet_line.png")
    fig.savefig(comparison_plot, dpi=150, bbox_inches="tight")
    plt.close(fig)

    summary_path = os.path.join(COMPARISON_ARTIFACTS_DIR, "summary.txt")
    with open(summary_path, "w", encoding="ascii") as summary_file:
        summary_file.write(f"Dry laptime: {dry_run['laptime']:.3f} s\n")
        summary_file.write(f"Wet-line laptime: {wet_run['laptime']:.3f} s\n")
        summary_file.write(
            f"Wet delta: {wet_run['laptime'] - dry_run['laptime']:.3f} s\n"
        )
        summary_file.write(
            f"Mean absolute wet-line shift from dry line: {np.mean(np.abs(wet_line_shift)):.3f} m\n"
        )
        summary_file.write(
            f"Max absolute wet-line shift from dry line: {np.max(np.abs(wet_line_shift)):.3f} m\n"
        )
        summary_file.write(f"Wet base grip multiplier: {WET_BASE_GRIP_MULTIPLIER:.3f}\n")
        summary_file.write(f"Dry line wet penalty: {WET_DRY_LINE_PENALTY:.3f}\n")
        summary_file.write(f"Dry line penalty width: {WET_DRY_LINE_WIDTH_M:.3f} m\n")
        summary_file.write(f"Min wet grip multiplier: {wet_grip.min():.3f}\n")
        summary_file.write(f"Max wet grip multiplier: {wet_grip.max():.3f}\n")

    return COMPARISON_ARTIFACTS_DIR, comparison_plot


def build_options(output_prefix):
    options = "<options>"
    options += "    <output_variables>"
    options += f"        <prefix>{output_prefix}</prefix>"
    options += "        <variables>"
    options += "            <laptime/>"
    options += "            <chassis.position.x/>"
    options += "            <chassis.position.y/>"
    options += "            <front-axle.steering-angle/>"
    options += "            <rear-axle.throttle/>"
    options += "            <chassis.velocity.x/>"
    options += "            <road.arclength/>"
    options += "            <road.lateral-displacement/>"
    options += "            <time/>"
    options += "            <chassis.attitude.yaw/>"
    options += "            <chassis.omega.z/>"
    options += "            <chassis.velocity.y/>"
    options += "            <rear-axle.left-tire.grip-multiplier/>"
    options += "        </variables>"
    options += "    </output_variables>"
    options += "    <initial_speed>30.0</initial_speed>"
    options += "    <print_level>5</print_level>"
    options += "</options>"

    return options


def run_optimal_laptime(vehicle_name, s, output_prefix):
    run = fastest_lap.download_variables(
        *fastest_lap.optimal_laptime(
            vehicle_name, TRACK_NAME, s, build_options(output_prefix)
        )
    )
    required_variables = (
        "laptime",
        "road.arclength",
        "road.lateral-displacement",
        "chassis.position.x",
        "chassis.position.y",
        "chassis.velocity.x",
    )
    missing_variables = [name for name in required_variables if name not in run]
    if missing_variables:
        available = ", ".join(sorted(run.keys())) or "none"
        raise RuntimeError(
            f"Missing optimal-laptime outputs for {vehicle_name}: "
            f"{', '.join(missing_variables)}. Available outputs: {available}"
        )

    return run


def main():
    fastest_lap.create_vehicle_from_xml(
        DRY_VEHICLE_NAME,
        "../../../../database/vehicles/kart/sodi-sr4-390cc.xml",
    )

    fastest_lap.create_vehicle_from_xml(
        WET_VEHICLE_NAME,
        "../../../../database/vehicles/kart/sodi-sr4-390cc.xml",
    )

    fastest_lap.create_track_from_xml(
        TRACK_NAME,
        "../../../../database/tracks/plesse_international_karting/plesse_internationale_karting.xml",
    )

    s = fastest_lap.track_download_data(TRACK_NAME, "arclength")

    dry_run = run_optimal_laptime(DRY_VEHICLE_NAME, s, "dry/")
    fastest_lap.vehicle_set_wet_surface(
        WET_VEHICLE_NAME,
        WET_BASE_GRIP_MULTIPLIER,
        WET_DRY_LINE_PENALTY,
        WET_DRY_LINE_WIDTH_M,
        dry_run["road.arclength"],
        dry_run["road.lateral-displacement"],
    )
    wet_run = run_optimal_laptime(WET_VEHICLE_NAME, s, "wet/")

    artifacts_dir, video_path = save_visualizations(
        dry_run,
        ARTIFACTS_DIR,
        "Sodi SR4 390cc at Plesse Internationale - dry line",
        "plesse-internationale_sodi-sr4_dry.mp4",
    )
    wet_artifacts_dir, wet_video_path = save_visualizations(
        wet_run,
        WET_ARTIFACTS_DIR,
        "Sodi SR4 390cc at Plesse Internationale - wet line",
        "plesse-internationale_sodi-sr4_wet-line.mp4",
        make_video=False,
    )
    comparison_dir, comparison_plot = save_comparison(dry_run, wet_run)

    print(f"Dry laptime: {dry_run['laptime']:.3f} s")
    print(f"Wet-line laptime: {wet_run['laptime']:.3f} s")
    print(f"Wet delta: {wet_run['laptime'] - dry_run['laptime']:.3f} s")
    print(f"Dry max speed: {max(dry_run['chassis.velocity.x']) * 3.6:.2f} km/h")
    print(f"Wet max speed: {max(wet_run['chassis.velocity.x']) * 3.6:.2f} km/h")
    print(f"Track points: {len(dry_run['road.arclength'])}")
    print(f"Dry visualizations: {artifacts_dir}")
    print(f"Wet visualizations: {wet_artifacts_dir}")
    print(f"Comparison: {comparison_dir}")
    print(f"Comparison plot: {comparison_plot}")
    print(f"Realtime MP4: {video_path}")
    if wet_video_path is not None:
        print(f"Wet realtime MP4: {wet_video_path}")


if __name__ == "__main__":
    main()

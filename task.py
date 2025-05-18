import numpy as np
import pandas as pd
import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.interpolate import interp1d
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional
import sys
from collections import defaultdict


@dataclass
class DroneMission:
    mission_id: int
    waypoints: np.ndarray
    time_at_waypoints: np.ndarray
    time_window: Tuple[float, float]
    max_speed: float
    priority: int
    is_3d: bool
    color: str
    label: str


@dataclass
class ConflictReport:
    status: str  # "clear" or "conflict detected"
    total_conflicts: int
    conflict_details: List[Dict]
    spatial_violations: int
    temporal_violations: int
    flight_advice: str


class DroneDeconflictionSystem:
    def __init__(self):
        self.fig = plt.figure(figsize=(14, 10))
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.safety_buffer = 15
        self.conflicts = []
        self.drones = []
        self.advisory_text = None
        self.primary_mission = None
        self.conflict_report = None

    def load_missions(self, csv_file: str) -> List[DroneMission]:
        """Load missions from CSV with the specified format"""
        try:
            data = pd.read_csv(csv_file)
            required_cols = ['mission_id', 'type', 'waypoints', 'time_at_waypoints',
                             'time_window', 'max_speed', 'priority']
            if not all(col in data.columns for col in required_cols):
                raise ValueError(f"CSV missing required columns: {required_cols}")

            missions = []
            for _, row in data.iterrows():
                try:
                    # Parse waypoints
                    wp_str = str(row['waypoints']).strip('"[]')
                    waypoints = np.array([
                        list(map(float, filter(None, point.split(','))))
                        for point in wp_str.split(';') if point.strip()
                    ])

                    # Parse times
                    times_str = str(row['time_at_waypoints']).strip('"')
                    times = np.array(list(map(float, filter(None, times_str.split(';')))))

                    # Parse time window
                    window_str = str(row['time_window']).strip('"')
                    time_window = tuple(map(float, filter(None, window_str.split(','))))

                    is_3d = waypoints.shape[1] == 3
                    mission_type = str(row['type']).lower().strip()
                    mission_id = int(row['mission_id'])
                    max_speed = float(row['max_speed'])
                    priority = int(row['priority'])

                    color = self._get_color_for_priority(priority)

                    missions.append(DroneMission(
                        mission_id=mission_id,
                        waypoints=waypoints,
                        time_at_waypoints=times,
                        time_window=time_window,
                        max_speed=max_speed,
                        priority=priority,
                        is_3d=is_3d,
                        color=color,
                        label=f"{mission_type}_{mission_id}" if mission_type != 'primary' else 'Primary'
                    ))
                except Exception as e:
                    print(f"⚠️ Error processing mission {row.get('mission_id', 'unknown')}: {e}")
                    continue

            if not missions:
                raise ValueError("No valid missions found")

            return missions

        except Exception as e:
            print(f"❌ Failed to load CSV: {e}")
            print("\n📋 Required CSV format:")
            print("mission_id,type,waypoints,time_at_waypoints,time_window,max_speed,priority")
            print('1,primary,"0,0,0;50,0,10;50,50,20","0;10;20","0,30",10,1')
            print('2,drone,"0,50,0;50,50,15;50,0,30","0;15;30","0,30",8,2')
            sys.exit(1)

    def _get_color_for_priority(self, priority: int) -> str:
        """Assign colors based on priority"""
        colors = {
            1: 'blue',  # Highest priority
            2: 'green',  # High priority
            3: 'orange',  # Medium priority
            4: 'red',  # Low priority
            5: 'purple'  # Very low priority
        }
        return colors.get(priority, 'gray')

    def run(self, csv_file: str, safety_buffer: float = 15) -> ConflictReport:
        """Run complete simulation and return conflict report"""
        try:
            print("🚀 Starting drone deconfliction system...")
            self.safety_buffer = safety_buffer

            # Load and validate missions
            missions = self.load_missions(csv_file)
            self.primary_mission = next((m for m in missions if m.label == 'Primary'), None)
            if not self.primary_mission:
                raise ValueError("No primary mission found")

            self.drones = [m for m in missions if m != self.primary_mission]
            print(f"✅ Loaded {len(self.drones)} drone missions")

            # Process trajectories
            primary_traj, primary_times = self._process_trajectory(self.primary_mission)
            drone_trajs, drone_times = zip(*[self._process_trajectory(d) for d in self.drones])

            # Setup visualization
            self._setup_visualization(self.primary_mission, self.drones)

            # Add text box for flight advisories
            self.advisory_text = self.ax.text2D(0.02, 0.95, "", transform=self.ax.transAxes,
                                                bbox=dict(facecolor='white', alpha=0.8))

            # Run comprehensive conflict analysis
            self._run_comprehensive_checks(primary_traj, primary_times, drone_trajs, drone_times)

            # Generate final flight advice
            self._generate_flight_advice()

            # Run animation
            self._animate(primary_traj, primary_times, drone_trajs, drone_times)

            return self.conflict_report

        except Exception as e:
            print(f"💥 Simulation failed: {e}")
            sys.exit(1)

    def _process_trajectory(self, mission, num_points=300):
        """Create smooth trajectory with boundary condition handling"""
        try:
            # Calculate segment distances and times
            segments = mission.waypoints[1:] - mission.waypoints[:-1]
            segment_distances = np.linalg.norm(segments, axis=1)
            segment_times = segment_distances / mission.max_speed

            # Handle single-point case
            if len(mission.time_at_waypoints) == 1:
                return np.tile(mission.waypoints, (num_points, 1)), \
                    np.linspace(mission.time_window[0], mission.time_window[1], num_points)

            # Calculate cumulative times with boundary checks
            adjusted_times = np.concatenate(([0], np.cumsum(segment_times)))

            # Normalize to fit original time window
            total_time = mission.time_window[1] - mission.time_window[0]
            time_scale = total_time / adjusted_times[-1] if adjusted_times[-1] > 0 else 1
            adjusted_times *= time_scale

            # Create interpolation with boundary condition handling
            norm_times = adjusted_times / adjusted_times[-1]

            # Ensure enough points for interpolation
            if len(norm_times) < 3:
                return mission.waypoints, mission.time_at_waypoints

            traj = np.column_stack([
                interp1d(norm_times, mission.waypoints[:, dim],
                         kind='quadratic', fill_value='extrapolate')(np.linspace(0, 1, num_points))
                for dim in range(mission.waypoints.shape[1])
            ])

            times = mission.time_window[0] + adjusted_times[-1] * np.linspace(0, 1, num_points)
            return traj, times

        except Exception as e:
            print(f"⚠️ Trajectory processing error for {mission.label}: {e}")
            # Fallback to linear interpolation
            return mission.waypoints, mission.time_at_waypoints

    def _setup_visualization(self, primary: DroneMission, drones: List[DroneMission]):
        """Initialize visualization with priority coloring"""
        self.fig.clf()
        self.ax = self.fig.add_subplot(111, projection='3d' if primary.is_3d else None)
        self.ax.grid(True)

        # Plot area
        all_waypoints = np.vstack([primary.waypoints] + [d.waypoints for d in drones])
        margin = 0.2 * np.ptp(all_waypoints, axis=0)
        self.ax.set_xlim([np.min(all_waypoints[:, 0]) - margin[0], np.max(all_waypoints[:, 0]) + margin[0]])
        self.ax.set_ylim([np.min(all_waypoints[:, 1]) - margin[1], np.max(all_waypoints[:, 1]) + margin[1]])
        if primary.is_3d:
            self.ax.set_zlim([np.min(all_waypoints[:, 2]) - margin[2], np.max(all_waypoints[:, 2]) + margin[2]])

        # Plot trajectories
        z_vals = primary.waypoints[:, 2] if primary.is_3d else np.zeros(len(primary.waypoints))
        self.ax.plot(primary.waypoints[:, 0], primary.waypoints[:, 1], z_vals,
                     'b-', linewidth=2, label='Primary Path')

        for drone in drones:
            z_vals = drone.waypoints[:, 2] if drone.is_3d else np.zeros(len(drone.waypoints))
            self.ax.plot(drone.waypoints[:, 0], drone.waypoints[:, 1], z_vals,
                         '--', color=drone.color, linewidth=1, label=f'{drone.label} (P:{drone.priority})')

        # Labels and legend
        self.ax.set_xlabel('X (m)')
        self.ax.set_ylabel('Y (m)')
        if primary.is_3d:
            self.ax.set_zlabel('Z (m)')
        self.ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()

    def _run_comprehensive_checks(self, primary_traj: np.ndarray, primary_times: np.ndarray,
                                  drone_trajs: List[np.ndarray], drone_times: List[np.ndarray]):
        """Run spatial and temporal conflict checks"""
        print("\n🔍 Running comprehensive conflict analysis...")

        # Create time points for analysis
        all_times = np.unique(np.sort(np.concatenate([
            primary_times,
            *[t for t in drone_times]
        ])))

        spatial_violations = 0
        temporal_violations = 0
        conflict_details = []

        # Check each time point
        for t in all_times:
            primary_pos = self._get_position(primary_traj, primary_times, t)

            for i, (drone_traj, drone_time, drone) in enumerate(zip(drone_trajs, drone_times, self.drones)):
                if t < drone_time[0] or t > drone_time[-1]:
                    continue  # Drone not active at this time

                drone_pos = self._get_position(drone_traj, drone_time, t)
                distance = np.linalg.norm(primary_pos - drone_pos)

                # Spatial check (minimum distance violation)
                if distance < self.safety_buffer:
                    spatial_violations += 1

                    # Temporal check (overlapping active periods)
                    primary_active = (t >= primary_times[0]) and (t <= primary_times[-1])
                    drone_active = (t >= drone_time[0]) and (t <= drone_time[-1])

                    if primary_active and drone_active:
                        temporal_violations += 1

                        conflict_details.append({
                            'time': t,
                            'location': primary_pos,
                            'distance': distance,
                            'drone_id': drone.mission_id,
                            'drone_priority': drone.priority,
                            'drone_label': drone.label,
                            'type': 'spatial_temporal',
                            'safety_buffer': self.safety_buffer
                        })

        # Create conflict report
        status = "clear" if not conflict_details else "conflict detected"
        self.conflict_report = ConflictReport(
            status=status,
            total_conflicts=len(conflict_details),
            conflict_details=sorted(conflict_details, key=lambda x: x['time']),
            spatial_violations=spatial_violations,
            temporal_violations=temporal_violations,
            flight_advice=""
        )

        print(f"✅ Analysis complete - {status.upper()}")
        print(f"• Spatial violations: {spatial_violations}")
        print(f"• Temporal violations: {temporal_violations}")
        print(f"• Total conflicts: {len(conflict_details)}")

    def _generate_flight_advice(self):
        """Generate comprehensive flight advice based on conflict analysis"""
        if not self.conflict_report or self.conflict_report.status == "clear":
            self.conflict_report.flight_advice = (
                "FLIGHT ADVISORY\n"
                "────────────────\n"
                "✅ Mission is CLEAR of conflicts\n"
                "No spatial or temporal violations detected\n"
                "Recommended actions:\n"
                "• Proceed with mission as planned\n"
                "• Maintain standard safety protocols"
            )
            return

        report = self.conflict_report
        conflict_count = report.total_conflicts
        worst_conflict = min(report.conflict_details, key=lambda x: x['distance'])

        # Base advisory
        advisory = [
            "FLIGHT ADVISORY",
            "────────────────",
            f"🚨 CONFLICT DETECTED ({conflict_count} incidents)",
            "",
            "WORST CASE INCIDENT:",
            f"• Time: {worst_conflict['time']:.1f}s",
            f"• Location: {tuple(np.round(worst_conflict['location'], 1))}",
            f"• Distance: {worst_conflict['distance']:.1f}m (Buffer: {worst_conflict['safety_buffer']:.1f}m)",
            f"• With: {worst_conflict['drone_label']} (Priority: {worst_conflict['drone_priority']})",
            "",
            "ANALYSIS SUMMARY:",
            f"• Spatial violations: {report.spatial_violations}",
            f"• Temporal violations: {report.temporal_violations}",
            "",
            "RECOMMENDED ACTIONS:"
        ]

        # Add priority-specific advice
        if worst_conflict['drone_priority'] > self.primary_mission.priority:
            advisory.extend([
                "• You have RIGHT OF WAY (higher priority)",
                "• Other drone should yield",
                "• Proceed with caution",
                "• Monitor conflict areas closely"
            ])
        elif worst_conflict['drone_priority'] == self.primary_mission.priority:
            advisory.extend([
                "⚠️ EQUAL PRIORITY CONFLICT",
                "• Coordinate with other drone operator",
                "• Consider altitude adjustment (+5m recommended)",
                "• Reduce speed by 20% in conflict zones",
                "• Establish communication protocol"
            ])
        else:
            advisory.extend([
                "🚨 YIELD REQUIRED (lower priority)",
                "• Immediate course correction needed",
                "• Altitude change of +10m recommended",
                "• Reduce speed by 30%",
                "• Prepare emergency procedures"
            ])

        # Add general advice based on conflict count
        if conflict_count > 5:
            advisory.extend([
                "",
                "⚠️ HIGH CONFLICT DENSITY WARNING:",
                "• Consider complete route replanning",
                "• Schedule alternative time window",
                "• Request airspace priority clearance"
            ])
        elif conflict_count > 2:
            advisory.extend([
                "",
                "⚠️ MULTIPLE CONFLICTS DETECTED:",
                "• Review all conflict points carefully",
                "• Consider partial route adjustments",
                "• Increase monitoring in conflict zones"
            ])

        self.conflict_report.flight_advice = "\n".join(advisory)

    def _animate(self, primary_traj: np.ndarray, primary_times: np.ndarray,
                 drone_trajs: List[np.ndarray], drone_times: List[np.ndarray]):
        """Run the animation with conflict detection"""
        # Create time points
        all_times = np.unique(np.concatenate([
            primary_times,
            *[t for t in drone_times]
        ]))

        # Initialize markers
        self.primary_marker, = self.ax.plot([], [], [], 'bo', markersize=10, label='Primary')
        self.drone_markers = [
            self.ax.plot([], [], [], 'o', color=d.color, markersize=8,
                         label=f'{d.label} (P:{d.priority})')[0]
            for d in self.drones
        ]
        self.conflict_markers = []

        def update(frame):
            try:
                t = all_times[frame]

                # Update primary position
                primary_pos = self._get_position(primary_traj, primary_times, t)
                self.primary_marker.set_data([primary_pos[0]], [primary_pos[1]])
                if primary_pos.size > 2:
                    self.primary_marker.set_3d_properties([primary_pos[2]])

                # Update drone positions
                for i, (drone_traj, drone_time) in enumerate(zip(drone_trajs, drone_times)):
                    drone_pos = self._get_position(drone_traj, drone_time, t)
                    self.drone_markers[i].set_data([drone_pos[0]], [drone_pos[1]])
                    if drone_pos.size > 2:
                        self.drone_markers[i].set_3d_properties([drone_pos[2]])

                # Clear previous conflict markers
                for marker in self.conflict_markers:
                    marker.remove()
                self.conflict_markers = []

                # Check for conflicts at current time
                current_conflicts = []
                for conflict in self.conflict_report.conflict_details:
                    if abs(conflict['time'] - t) < 0.1:  # Fuzzy time match
                        current_conflicts.append(conflict)
                        marker = self.ax.scatter(*conflict['location'][:3], c='red',
                                                 s=100, alpha=0.7, marker='x')
                        self.conflict_markers.append(marker)

                # Update advisory text
                if current_conflicts:
                    self._update_realtime_advisory(t, current_conflicts)
                else:
                    self.advisory_text.set_text(
                        f"FLIGHT STATUS\n────────────────\n"
                        f"Time: {t:.1f}s\n"
                        f"Status: CLEAR\n"
                        f"Next conflict in: {self._next_conflict_time(t):.1f}s"
                    )

                self.ax.set_title(f'Time: {t:.1f}s | Conflicts: {len(current_conflicts)}')
                return [self.primary_marker] + self.drone_markers + self.conflict_markers
            except Exception as e:
                print(f"⚠️ Animation error: {e}")
                return []

        # Create faster animation (25ms interval)
        try:
            self.ani = FuncAnimation(
                self.fig, update, frames=len(all_times),
                interval=25, blit=False, repeat=False
            )
            plt.show()
        except Exception as e:
            print(f"💥 Animation failed: {e}")

    def _next_conflict_time(self, current_time: float) -> float:
        """Find time until next conflict"""
        if not self.conflict_report.conflict_details:
            return float('inf')

        future_conflicts = [c for c in self.conflict_report.conflict_details
                            if c['time'] > current_time]

        if not future_conflicts:
            return float('inf')

        return min(c['time'] - current_time for c in future_conflicts)

    def _update_realtime_advisory(self, t: float, current_conflicts: List[dict]):
        """Update real-time advisory during animation"""
        worst_conflict = min(current_conflicts, key=lambda x: x['distance'])

        advisory = [
            "FLIGHT ADVISORY",
            "────────────────",
            f"Time: {t:.1f}s",
            f"Location: {tuple(np.round(worst_conflict['location'], 1))}",
            f"Distance: {worst_conflict['distance']:.1f}m",
            f"With: {worst_conflict['drone_label']} (P:{worst_conflict['drone_priority']})",
            "",
            "IMMEDIATE ACTION:"
        ]

        if worst_conflict['drone_priority'] > self.primary_mission.priority:
            advisory.append("PROCEED: You have right of way")
        elif worst_conflict['drone_priority'] == self.primary_mission.priority:
            advisory.append("CAUTION: Equal priority conflict")
        else:
            advisory.append("WARNING: Yield to higher priority drone")

        self.advisory_text.set_text("\n".join(advisory))

    def check_mission(self, primary_mission: DroneMission, drone_missions: List[DroneMission]) -> ConflictReport:
        """Query interface for mission checking"""
        print("\n🔍 Running mission conflict check...")

        # Process trajectories
        primary_traj, primary_times = self._process_trajectory(primary_mission)
        drone_trajs, drone_times = zip(*[self._process_trajectory(d) for d in drone_missions])

        # Run comprehensive checks
        self._run_comprehensive_checks(primary_traj, primary_times, drone_trajs, drone_times)
        self._generate_flight_advice()

        return self.conflict_report

    def _get_position(self, trajectory: np.ndarray, times: np.ndarray, t: float) -> np.ndarray:
        """Get position at time t"""
        idx = np.argmin(np.abs(times - t))
        pos = trajectory[idx]
        # Ensure we always return at least 2D coordinates
        if pos.size < 2:
            return np.append(pos, [0])
        return pos


if __name__ == "__main__":
    try:
        if len(sys.argv) > 1:
            csv_file = sys.argv[1]
        else:
            csv_file = "mid conflict.csv"

        system = DroneDeconflictionSystem()
        report = system.run(csv_file, safety_buffer=15)

        # Print final report
        print("\n📝 FINAL MISSION REPORT")
        print("=" * 40)
        print(report.flight_advice)
        print("\nConflict Details:")
        for i, conflict in enumerate(report.conflict_details[:5]):  # Show first 5 conflicts
            print(f"\nConflict {i + 1}:")
            print(f"• Time: {conflict['time']:.1f}s")
            print(f"• Location: {tuple(np.round(conflict['location'], 1))}")
            print(f"• Distance: {conflict['distance']:.1f}m")
            print(f"• With: {conflict['drone_label']} (Priority: {conflict['drone_priority']})")
            print(f"• Type: {conflict['type']}")

        if report.total_conflicts > 5:
            print(f"\n... and {report.total_conflicts - 5} more conflicts")

    except KeyboardInterrupt:
        print("\n🛑 Simulation stopped by user")
        sys.exit(0)
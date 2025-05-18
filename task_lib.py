import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpec
from PyQt5.QtWidgets import (QApplication, QMainWindow, QVBoxLayout, QHBoxLayout,
                             QWidget, QTableWidget, QTableWidgetItem, QLabel,
                             QTextEdit, QPushButton, QGroupBox)
from rtree import index
import sys

# ==============================================
# 1. Built-in Mission Data
# ==============================================

MISSIONS_DATA = {
    "primary": {
        "id": 1,
        "waypoints": [
            {"x": 0, "y": 0, "z": 0, "t": 0},
            {"x": 100, "y": 0, "z": 20, "t": 20},
            {"x": 100, "y": 100, "z": 40, "t": 40},
            {"x": 0, "y": 100, "z": 60, "t": 60},
            {"x": 0, "y": 0, "z": 80, "t": 80}
        ],
        "priority": 1
    },
    "others": [
        {
            "id": 2,
            "waypoints": [
                {"x": 0, "y": 100, "z": 0, "t": 0},
                {"x": 100, "y": 100, "z": 20, "t": 25},
                {"x": 100, "y": 0, "z": 40, "t": 50},
                {"x": 0, "y": 0, "z": 60, "t": 75},
                {"x": 0, "y": 100, "z": 80, "t": 100}
            ],
            "priority": 2
        },
        {
            "id": 3,
            "waypoints": [
                {"x": 50, "y": -50, "z": 10, "t": 0},
                {"x": 50, "y": 150, "z": 30, "t": 30},
                {"x": 150, "y": 150, "z": 50, "t": 60},
                {"x": 150, "y": -50, "z": 70, "t": 90},
                {"x": 50, "y": -50, "z": 90, "t": 120}
            ],
            "priority": 3
        }
    ]
}


# ==============================================
# 2. Enhanced Deconfliction Engine with Advice
# ==============================================

class DeconflictionEngine:
    def __init__(self, safety_buffer=10.0, time_threshold=5.0):
        self.safety_buffer = safety_buffer
        self.time_threshold = time_threshold
        self.spatial_index = index.Index(properties=index.Property(dimension=3))

    def build_indices(self, df):
        for idx, row in df.iterrows():
            self.spatial_index.insert(
                idx,
                (row['x'] - self.safety_buffer,
                 row['y'] - self.safety_buffer,
                 row['z'] - self.safety_buffer,
                 row['x'] + self.safety_buffer,
                 row['y'] + self.safety_buffer,
                 row['z'] + self.safety_buffer)
            )

    def check_conflicts(self, primary_df, other_drones_df):
        conflicts = []
        resolution_advice = []

        for _, primary_point in primary_df.iterrows():
            bbox = (
                primary_point['x'] - self.safety_buffer,
                primary_point['y'] - self.safety_buffer,
                primary_point['z'] - self.safety_buffer,
                primary_point['x'] + self.safety_buffer,
                primary_point['y'] + self.safety_buffer,
                primary_point['z'] + self.safety_buffer
            )

            nearby_indices = list(self.spatial_index.intersection(bbox))

            for idx in nearby_indices:
                other_point = other_drones_df.iloc[idx]
                time_diff = abs(primary_point['t'] - other_point['t'])

                if time_diff < self.time_threshold:
                    conflict = {
                        'x': primary_point['x'],
                        'y': primary_point['y'],
                        'z': primary_point['z'],
                        't': primary_point['t'],
                        'conflict_drone': other_point['drone_id'],
                        'time_diff': time_diff,
                        'other_priority': other_point.get('priority', 2)
                    }
                    conflicts.append(conflict)

                    # Generate resolution advice
                    advice = self.generate_advice(primary_point, other_point)
                    resolution_advice.append(advice)

        return conflicts, resolution_advice

    def generate_advice(self, primary_point, other_point):
        """Generate resolution advice for each conflict"""
        priority_diff = 1 - other_point.get('priority', 2)  # Primary has priority 1

        if priority_diff < 0:
            action = "YIELD: Lower priority drone should yield"
            resolution = f"Drone {other_point['drone_id']} should adjust trajectory"
        else:
            action = "PROCEED: You have right of way"
            resolution = "Maintain course but monitor situation"

        vertical_sep = abs(primary_point['z'] - other_point['z'])
        horizontal_sep = np.sqrt((primary_point['x'] - other_point['x']) ** 2 +
                                 (primary_point['y'] - other_point['y']) ** 2)

        suggestions = []
        if vertical_sep < self.safety_buffer:
            suggestions.append(f"Adjust altitude by +{self.safety_buffer}m")
        if horizontal_sep < self.safety_buffer:
            suggestions.append(f"Change heading by 15° right")
            suggestions.append(f"Reduce speed by 20%")

        if not suggestions:
            suggestions.append("No immediate action required but maintain vigilance")

        return {
            'time': primary_point['t'],
            'location': f"({primary_point['x']:.1f}, {primary_point['y']:.1f}, {primary_point['z']:.1f})",
            'conflict_with': f"Drone {other_point['drone_id']}",
            'action': action,
            'suggestions': suggestions,
            'priority_comparison': f"Primary priority: 1 vs Drone {other_point['drone_id']} priority: {other_point.get('priority', 2)}"
        }


# ==============================================
# 3. Four-Quadrant Dashboard with Advice Panel
# ==============================================

class UAVDashboard(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("UAV Deconfliction Dashboard with Flight Advice")
        self.setGeometry(100, 100, 1400, 1000)

        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)

        # Left side: Visualizations (3 quadrants)
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)

        # Create matplotlib figure with 3 subplots
        self.fig = Figure(figsize=(10, 8))
        self.canvas = FigureCanvasQTAgg(self.fig)
        left_layout.addWidget(self.canvas)

        # Create 2x2 grid (we'll use 3 quadrants)
        gs = GridSpec(2, 2, figure=self.fig)

        # Quadrant 1: 3D Trajectory Plot
        self.ax_3d = self.fig.add_subplot(gs[0, 0], projection='3d')
        self.ax_3d.set_title("3D Trajectories")

        # Quadrant 2: Spatial Check (XY View)
        self.ax_xy = self.fig.add_subplot(gs[0, 1])
        self.ax_xy.set_title("XY View")

        # Quadrant 3: Temporal Check (Time View)
        self.ax_time = self.fig.add_subplot(gs[1, :])
        self.ax_time.set_title("Time vs Position")

        # Right side: Control and advice panels
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)

        # Conflict Table (Quadrant 4)
        self.conflict_table = QTableWidget()
        self.conflict_table.setColumnCount(5)
        self.conflict_table.setHorizontalHeaderLabels(['Time', 'X', 'Y', 'Z', 'Drone ID'])
        right_layout.addWidget(QLabel("Conflict Details"))
        right_layout.addWidget(self.conflict_table)

        # Flight Advice Panel
        advice_group = QGroupBox("Flight Advisory")
        advice_layout = QVBoxLayout()

        self.advice_text = QTextEdit()
        self.advice_text.setReadOnly(True)
        self.advice_text.setStyleSheet("font-size: 12pt;")

        self.action_buttons = QWidget()
        button_layout = QHBoxLayout()

        self.btn_altitude = QPushButton("Adjust Altitude")
        self.btn_heading = QPushButton("Change Heading")
        self.btn_speed = QPushButton("Reduce Speed")
        self.btn_proceed = QPushButton("Proceed with Caution")

        button_layout.addWidget(self.btn_altitude)
        button_layout.addWidget(self.btn_heading)
        button_layout.addWidget(self.btn_speed)
        button_layout.addWidget(self.btn_proceed)

        self.action_buttons.setLayout(button_layout)

        advice_layout.addWidget(self.advice_text)
        advice_layout.addWidget(self.action_buttons)
        advice_group.setLayout(advice_layout)

        right_layout.addWidget(advice_group)

        # Add widgets to main layout
        main_layout.addWidget(left_widget, 70)  # 70% width
        main_layout.addWidget(right_widget, 30)  # 30% width

        # Initialize data and engine
        self.df = self.create_dataframe()
        self.primary_df = self.df[self.df['type'] == 'primary']
        self.other_drones_df = self.df[self.df['type'] == 'other']

        self.engine = DeconflictionEngine()
        self.engine.build_indices(self.other_drones_df)

        # Connect buttons
        self.btn_altitude.clicked.connect(self.suggest_altitude_adjustment)
        self.btn_heading.clicked.connect(self.suggest_heading_change)
        self.btn_speed.clicked.connect(self.suggest_speed_reduction)
        self.btn_proceed.clicked.connect(self.suggest_proceed_with_caution)

        # Run initial analysis
        self.run_analysis()

    def create_dataframe(self):
        """Convert built-in data to DataFrame with priorities"""
        records = []

        # Primary mission
        for wp in MISSIONS_DATA["primary"]["waypoints"]:
            records.append({
                "x": wp["x"],
                "y": wp["y"],
                "z": wp["z"],
                "t": wp["t"],
                "drone_id": MISSIONS_DATA["primary"]["id"],
                "type": "primary",
                "priority": MISSIONS_DATA["primary"]["priority"]
            })

        # Other drones
        for drone in MISSIONS_DATA["others"]:
            for wp in drone["waypoints"]:
                records.append({
                    "x": wp["x"],
                    "y": wp["y"],
                    "z": wp["z"],
                    "t": wp["t"],
                    "drone_id": drone["id"],
                    "type": "other",
                    "priority": drone["priority"]
                })

        return pd.DataFrame(records)

    def run_analysis(self):
        """Run deconfliction checks and update all displays"""
        conflicts, advice_list = self.engine.check_conflicts(self.primary_df, self.other_drones_df)

        # Update visualizations
        self.update_3d_plot(conflicts)
        self.update_xy_plot(conflicts)
        self.update_time_plot(conflicts)
        self.update_conflict_table(conflicts)

        # Update advice panel
        self.update_advice_panel(advice_list)

        # Redraw canvas
        self.canvas.draw()

    def update_advice_panel(self, advice_list):
        """Update the flight advice panel"""
        if not advice_list:
            self.advice_text.setHtml("<h3>No conflicts detected</h3><p>Flight path is clear. Proceed as planned.</p>")
            return

        html_content = "<h3>Flight Advisory</h3>"

        for advice in advice_list:
            html_content += f"""
            <div style='margin-bottom: 15px; border-bottom: 1px solid #ccc; padding-bottom: 10px;'>
                <p><b>Time:</b> {advice['time']}</p>
                <p><b>Location:</b> {advice['location']}</p>
                <p><b>Conflict with:</b> {advice['conflict_with']}</p>
                <p><b>Priority:</b> {advice['priority_comparison']}</p>
                <p style='color: {'red' if 'YIELD' in advice['action'] else 'green'};'>
                    <b>Action:</b> {advice['action']}
                </p>
                <p><b>Suggested resolutions:</b></p>
                <ul>
                    {''.join(f'<li>{suggestion}</li>' for suggestion in advice['suggestions'])}
                </ul>
            </div>
            """

        self.advice_text.setHtml(html_content)

    # [Previous visualization methods remain the same...]
    def update_3d_plot(self, conflicts):
        self.ax_3d.clear()
        colors = plt.cm.tab10(np.linspace(0, 1, len(self.df['drone_id'].unique())))

        for drone_id, group in self.df.groupby('drone_id'):
            color = colors[int(drone_id) % len(colors)]
            label = 'Primary' if group['type'].iloc[0] == 'primary' else f'Drone {drone_id}'
            self.ax_3d.plot(
                group['x'], group['y'], group['z'],
                label=label, color=color, alpha=0.6
            )

        if conflicts:
            conflict_points = np.array([[c['x'], c['y'], c['z']] for c in conflicts])
            self.ax_3d.scatter(
                conflict_points[:, 0], conflict_points[:, 1], conflict_points[:, 2],
                color='red', s=100, marker='x', label='Conflicts'
            )

        self.ax_3d.set_xlabel('X (m)')
        self.ax_3d.set_ylabel('Y (m)')
        self.ax_3d.set_zlabel('Z (m)')
        self.ax_3d.legend()

    def update_xy_plot(self, conflicts):
        self.ax_xy.clear()
        colors = plt.cm.tab10(np.linspace(0, 1, len(self.df['drone_id'].unique())))

        for drone_id, group in self.df.groupby('drone_id'):
            color = colors[int(drone_id) % len(colors)]
            label = 'Primary' if group['type'].iloc[0] == 'primary' else f'Drone {drone_id}'
            self.ax_xy.plot(
                group['x'], group['y'],
                label=label, color=color, alpha=0.6
            )

        if conflicts:
            conflict_points = np.array([[c['x'], c['y']] for c in conflicts])
            self.ax_xy.scatter(
                conflict_points[:, 0], conflict_points[:, 1],
                color='red', s=100, marker='x', label='Conflicts'
            )

        self.ax_xy.set_title("XY View (Top-Down)")
        self.ax_xy.legend()

    def update_time_plot(self, conflicts):
        self.ax_time.clear()
        start_point = self.primary_df.iloc[0][['x', 'y', 'z']].values
        self.df['distance'] = self.df.apply(
            lambda row: np.linalg.norm([row['x'] - start_point[0],
                                        row['y'] - start_point[1],
                                        row['z'] - start_point[2]]),
            axis=1
        )

        colors = plt.cm.tab10(np.linspace(0, 1, len(self.df['drone_id'].unique())))

        for drone_id, group in self.df.groupby('drone_id'):
            color = colors[int(drone_id) % len(colors)]
            label = 'Primary' if group['type'].iloc[0] == 'primary' else f'Drone {drone_id}'
            self.ax_time.plot(
                group['t'], group['distance'],
                label=label, color=color, alpha=0.6
            )

        if conflicts:
            conflict_times = [c['t'] for c in conflicts]
            conflict_dists = [np.linalg.norm([c['x'] - start_point[0],
                                              c['y'] - start_point[1],
                                              c['z'] - start_point[2]])
                              for c in conflicts]
            self.ax_time.scatter(
                conflict_times, conflict_dists,
                color='red', s=100, marker='x', label='Conflicts'
            )

        self.ax_time.set_title("Time vs Distance from Start")
        self.ax_time.legend()

    def update_conflict_table(self, conflicts):
        self.conflict_table.setRowCount(len(conflicts))
        for i, conflict in enumerate(conflicts):
            self.conflict_table.setItem(i, 0, QTableWidgetItem(str(conflict['t'])))
            self.conflict_table.setItem(i, 1, QTableWidgetItem(str(conflict['x'])))
            self.conflict_table.setItem(i, 2, QTableWidgetItem(str(conflict['y'])))
            self.conflict_table.setItem(i, 3, QTableWidgetItem(str(conflict['z'])))
            self.conflict_table.setItem(i, 4, QTableWidgetItem(str(conflict['conflict_drone'])))

    # Action suggestion methods
    def suggest_altitude_adjustment(self):
        current_text = self.advice_text.toPlainText()
        self.advice_text.setHtml(f"""
            <h3>Altitude Adjustment Recommended</h3>
            <p>{current_text}</p>
            <p style='color: blue;'><b>Selected Action:</b> Increase altitude by 10m to avoid conflict</p>
            <p>This will create vertical separation from other traffic.</p>
        """)

    def suggest_heading_change(self):
        current_text = self.advice_text.toPlainText()
        self.advice_text.setHtml(f"""
            <h3>Heading Change Recommended</h3>
            <p>{current_text}</p>
            <p style='color: blue;'><b>Selected Action:</b> Adjust heading 15° right</p>
            <p>This will create horizontal separation from conflicting drone.</p>
        """)

    def suggest_speed_reduction(self):
        current_text = self.advice_text.toPlainText()
        self.advice_text.setHtml(f"""
            <h3>Speed Reduction Recommended</h3>
            <p>{current_text}</p>
            <p style='color: blue;'><b>Selected Action:</b> Reduce speed by 20%</p>
            <p>This will allow the other drone to pass through the conflict area first.</p>
        """)

    def suggest_proceed_with_caution(self):
        current_text = self.advice_text.toPlainText()
        self.advice_text.setHtml(f"""
            <h3>Proceeding with Caution</h3>
            <p>{current_text}</p>
            <p style='color: green;'><b>Selected Action:</b> Maintaining course with increased vigilance</p>
            <p>Monitor the situation closely as you have right of way.</p>
        """)


# ==============================================
# 4. Run the Application
# ==============================================

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = UAVDashboard()
    window.show()
    sys.exit(app.exec_())
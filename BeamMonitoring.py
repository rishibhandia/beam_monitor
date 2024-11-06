import cv2
import numpy as np
import traceback
from datetime import datetime
import matplotlib
matplotlib.use('Qt5Agg')  # Use PyQt5 as the backend
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.gridspec import GridSpec

from scipy.optimize import curve_fit
import pandas as pd
from pathlib import Path
import time
from threading import Thread, Lock
from queue import Queue
import sys
import logging
from typing import Optional, Tuple, List
import argparse

# Set up logging
# Set up logging once at the module level
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('beam_monitor.log'),
        logging.StreamHandler(sys.stdout)
    ]
)



class BeamMonitorException(Exception):
    """Base exception class for beam monitoring errors"""
    pass

class CameraError(BeamMonitorException):
    """Exception raised for camera-related errors"""
    pass

class FittingError(BeamMonitorException):
    """Exception raised for Gaussian fitting errors"""
    pass
class CameraSelector:
    def __init__(self):
        self.cameras: dict[int, str] = {}
        
    def get_camera_info(self, index: int) -> Optional[str]:
        """Get information about a specific camera"""
        try:
            cap = cv2.VideoCapture(index)
            if not cap.isOpened():
                return None
                
            # Try to get camera properties
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            # Try to get camera name (this might not work on all systems)
            try:
                # Attempt to get camera name - implementation varies by system
                name = cap.getBackendName()
            except:
                name = f"Camera {index}"
            
            # Take a test frame
            ret, frame = cap.read()
            if ret:
                info = f"{name} ({width}x{height})"
            else:
                info = f"{name} (No test frame)"
                
            cap.release()
            return info
        except CameraError as e:
            logger.warning(f"Error getting info for camera {index}: {e}")
            return None

    def find_cameras(self) -> dict[int, str]:
        """Find all available cameras and get their information"""
        self.cameras.clear()
        
        max_cameras = len(cv2.videoio_registry.getCameraBackends())
        print("Scanning for cameras...")
        for i in range(max_cameras):
            info = self.get_camera_info(i)
            if info:
                self.cameras[i] = info
                print(f"Found camera {i}: {info}")
        
        return self.cameras

    def select_camera(self) -> Optional[int]:
        """Interactive camera selection"""
        if not self.cameras:
            self.find_cameras()
            
        if not self.cameras:
            print("No cameras found!")
            return None
            
        while True:
            print("\nAvailable cameras:")
            for idx, info in self.cameras.items():
                print(f"{idx}: {info}")
            
            try:
                choice = input("\nSelect camera number (or 'q' to quit, 'r' to rescan): ")
                
                if choice.lower() == 'q':
                    return None
                elif choice.lower() == 'r':
                    self.find_cameras()
                    continue
                    
                choice = int(choice)
                if choice in self.cameras:
                    return choice
                else:
                    print("Invalid camera number!")
            except ValueError:
                print("Please enter a valid number!")

    def preview_camera(self, camera_index: int, duration: int = 5):
        """Show a preview of the selected camera"""
        try:
            cap = cv2.VideoCapture(camera_index)
            if not cap.isOpened():
                print("Could not open camera for preview")
                return False
                
            print(f"\nShowing camera preview for {duration} seconds...")
            print("Press 'q' to stop preview")
            
            start_time = time.time()
            while time.time() - start_time < duration:
                ret, frame = cap.read()
                if not ret:
                    break
                    
                cv2.imshow('Camera Preview', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                    
            cap.release()
            cv2.destroyAllWindows()
            
            # Ask for confirmation
            while True:
                choice = input("\nUse this camera? (y/n): ").lower()
                if choice in ['y', 'n']:
                    return choice == 'y'
                    
        except CameraError as e:
            logger.error(f"Error during camera preview: {e}")
            return False
class BeamAnalyzer:
    def __init__(self):
        """Initialize the 2D Gaussian fitting class"""
        self.lock = Lock()
        self.centers_queue = Queue(maxsize=100)
        self.logger = logging.getLogger(__name__)
        
    @staticmethod
    def gaussian_2d(xy, amplitude, x0, y0, sigma_x, sigma_y, offset):
        """2D Gaussian function"""
        x, y = xy
        exp_term = -((x - x0)**2 / (2 * sigma_x**2) + 
                     (y - y0)**2 / (2 * sigma_y**2))
        return offset + amplitude * np.exp(exp_term)

    def fit_gaussian(self, image: np.ndarray) -> Optional[dict]:
        """
        Fit a 2D Gaussian to the image
        
        Args:
            image: Input grayscale image as numpy array
            
        Returns:
            Dictionary containing fit parameters or None if fit fails
            
        Raises:
            FittingError: If the Gaussian fitting fails
        """
        try:
            # Validate input
            if image is None or image.size == 0:
                raise ValueError("Invalid image input")

            # Create x and y indices
            y, x = np.indices(image.shape)
            xy = (x.ravel(), y.ravel())
            
            # Initial guesses for parameters
            max_intensity = np.max(image)
            center_y, center_x = np.unravel_index(np.argmax(image), image.shape)
            
            initial_guess = [
                max_intensity,  # amplitude
                center_x,       # x0
                center_y,       # y0
                20.0,          # sigma_x
                20.0,          # sigma_y
                0.0            # offset
            ]
            
            # Perform the fit
            popt, pcov = curve_fit(self.gaussian_2d, xy, image.ravel(), 
                                 p0=initial_guess, maxfev=1000)
            
            # Validate fit results
            if not np.all(np.isfinite(popt)):
                raise FittingError("Fit resulted in non-finite parameters")
            
            return {
                'x': float(popt[1]), 
                'y': float(popt[2]),
                'sigma_x': float(popt[3]), 
                'sigma_y': float(popt[4]),
                'amplitude': float(popt[0]),
                'offset': float(popt[5])
            }
            
        except Exception as e:
            self.logger.error(f"Gaussian fitting failed: {str(e)}")
            raise FittingError(f"Failed to fit Gaussian: {str(e)}") from e
        
    

class BeamMonitor:
    def __init__(self, camera_index: int=0, 
                 interval=30,      # Interval in seconds for both saving and updating
                 plot_padding=10,  # Padding around scatter points
                 logging_time=24): # Time to log in hours
        """
        Initialize the beam monitoring system
        
        Args:
            camera_index (int): Index of the camera to use
            interval (float): Interval in seconds for saving data and updating plots
            plot_padding (float): Padding to add around scatter plot points
            logging_time (int): Time to log data in hours
        """
        self.camera_index = camera_index
        self.interval = interval
        self.plot_padding = plot_padding
        self.logging_time = logging_time
        
        # Calculate total number of data points to collect
        self.total_points = int((logging_time * 3600) / interval)
        self.points_collected = 0
        
        self.analyzer = BeamAnalyzer()
        self.running = False
        self.cap = None
        self.logger = logging.getLogger(__name__)
        
        # Log the monitoring parameters
        self.logger.info("Initializing beam monitor:")
        self.logger.info(f"Logging time: {logging_time} hours")
        self.logger.info(f"Interval: {interval} seconds")
        self.logger.info(f"Total data points to collect: {self.total_points}")
        
        # Create output directories with timestamp
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.base_dir = Path(f"beam_monitor_data_{timestamp}")
            self.image_dir = self.base_dir / "images"
            self.data_dir = self.base_dir / "data"
            self.image_dir.mkdir(parents=True, exist_ok=True)
            self.data_dir.mkdir(parents=True, exist_ok=True)
            
            # Save monitoring parameters
            self.save_parameters()
        except Exception as e:
            raise BeamMonitorException(f"Failed to create directories: {str(e)}")
        
        
        
        # Initialize data storage
        self.positions_df = pd.DataFrame({
            'timestamp': pd.Series(dtype='datetime64[ns]'),
            'x': pd.Series(dtype='float64'),
            'y': pd.Series(dtype='float64')
        })
        self.last_save_time = time.time()

    @staticmethod
    def list_available_cameras() -> list[int]:
        """
        List all available camera indices
        
        Args:
            max_cameras: Maximum number of cameras to check
            
        Returns:
            List of available camera indices
        """
        max_cameras = len(cv2.videoio_registry.getCameraBackends())
        available_cameras = []
        for i in range(max_cameras):
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                ret, _ = cap.read()
                if ret:
                    available_cameras.append(i)
                cap.release()
        return available_cameras
    

    def connect_camera(self) -> bool:
        """
        Attempt to connect to the camera
        
        Returns:
            bool: True if connection successful, False otherwise
        """
        try:
            if self.cap is not None:
                self.cap.release()
            
            self.cap = cv2.VideoCapture(self.camera_index)
            if not self.cap.isOpened():
                return False
            
            # Test frame capture
            ret, frame = self.cap.read()
            if not ret or frame is None:
                self.cap.release()
                return False
                
            return True
            
        except CameraError as e:
            self.logger.error(f"Camera connection error: {str(e)}")
            return False
    def _save_data(self, frame: np.ndarray, timestamp: datetime):
        """
        Save the current frame and update the data file
        
        Args:
            frame: Current camera frame
            timestamp: Current timestamp
        """
        try:
            # Save image
            image_filename = self.image_dir / f"beam_{timestamp:%Y%m%d_%H%M%S}.png"
            cv2.imwrite(str(image_filename), frame)
            
            # Save position data
            self._save_positions()
            
        except Exception as e:
            self.logger.error(f"Error saving data: {str(e)}")

    def _save_positions(self):
        """Save position data to CSV"""
        try:
            data_filename = self.data_dir / "beam_positions.csv"
            self.positions_df.to_csv(data_filename, index=False)
        except Exception as e:
            self.logger.error(f"Error saving positions data: {str(e)}")

    def _save_final_data(self):
        """Save final data before shutdown"""
        try:
            self._save_positions()
            self.logger.info("Final data saved successfully")
        except Exception as e:
            self.logger.error(f"Error saving final data: {str(e)}")

    def start(self):
        """
        Start the monitoring system
        
        Raises:
            CameraError: If camera initialization fails
        """
        # Check available cameras
        available_cameras = self.list_available_cameras()
        if not available_cameras:
            raise CameraError("No cameras available")
        
        if self.camera_index not in available_cameras:
            available_str = ", ".join(map(str, available_cameras))
            raise CameraError(
                f"Camera index {self.camera_index} not available. "
                f"Available cameras: {available_str}"
            )
        
        # Try to connect to camera
        if not self.connect_camera():
            raise CameraError(f"Failed to initialize camera {self.camera_index}")
        
        self.running = True
        self.logger.info("Starting beam monitor")
        
        # Start the monitoring thread
        self.monitor_thread = Thread(target=self._monitor_loop)
        self.monitor_thread.daemon = True  # Allow program to exit if thread is running
        self.monitor_thread.start()
        
        # Start the visualization
        self._start_visualization()

    def save_parameters(self):
        """Save monitoring parameters to a file"""
        params = {
            'logging_time_hours': self.logging_time,
            'interval_seconds': self.interval,
            'total_points': self.total_points,
            'start_time': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'camera_index': self.camera_index
        }
        
        param_file = self.base_dir / "monitoring_parameters.txt"
        with open(param_file, 'w') as f:
            for key, value in params.items():
                f.write(f"{key}: {value}\n")


    

    def _monitor_loop(self):
        """Main monitoring loop with error handling and recovery"""
        consecutive_failures = 0
        max_consecutive_failures = 5
        start_time = time.time()
        
        while self.running and self.points_collected < self.total_points:
            try:
                current_time = time.time()
                elapsed_hours = (current_time - start_time) / 3600
                
                # Log progress every min
                if self.points_collected % int(60 / self.interval) == 0:
                    self.logger.info(f"Logging progress: {elapsed_hours:.1f} hours elapsed, "
                                   f"{self.points_collected}/{self.total_points} points collected")
                
                if not self.cap.isOpened():
                    raise CameraError("Camera connection lost")
                    
                ret, frame = self.cap.read()
                if not ret or frame is None:
                    raise CameraError("Failed to capture frame")
                
                # Convert to grayscale
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                #check for saturation
                num_saturated_pixels = np.sum(gray >= 255)
                if num_saturated_pixels > 0:
                    self.logger.warning(f" {num_saturated_pixels} saturated pixels detected! out of {gray.size}")
                            
                # Analyze the beam
                result = self.analyzer.fit_gaussian(gray)

                
                
                # Reset failure counter on success
                consecutive_failures = 0
                
                timestamp = datetime.now()
                
                # Save data
                new_row = pd.DataFrame({
                    'timestamp': [timestamp],
                    'x': [result['x']],
                    'y': [result['y']]
                })
                self.positions_df = pd.concat([self.positions_df, new_row], 
                                           ignore_index=True)
                
                # Add to visualization queue
                self.analyzer.centers_queue.put((result['x'], result['y'],(current_time - start_time)))
                
                # Save data if interval has elapsed
                if current_time - self.last_save_time >= self.interval:
                    center_y = int(result["y"])
                    radius = 5  # Size of the point
                    center_x = int(result["x"])
                    color = (0, 0, 255)  # BGR format - this is red
                    thickness = -1  # -1 means filled circle

                    cv2.circle(frame, (center_x, center_y), radius, color, thickness)

                    self._save_data(frame, timestamp)
                    self.last_save_time = current_time
                    self.points_collected += 1
                
                    # Check if we've collected all points
                    if self.points_collected >= self.total_points:
                        self.logger.info("Data collection complete")
                        self.stop()
                        break
                
                # Sleep to maintain the specified interval
                time.sleep(max(0, self.interval - (time.time() - current_time)))
                    
            except CameraError as e:
                consecutive_failures += 1
                self.logger.error(f"Camera error: {str(e)}")
                
                if consecutive_failures >= max_consecutive_failures:
                    self.logger.error("Too many consecutive failures, attempting camera reconnection")
                    if not self.connect_camera():
                        self.logger.error("Camera reconnection failed, waiting before retry")
                        time.sleep(self.interval)
                        
            except Exception as e:
                self.logger.warning(f"Fitting error: {str(e)}")
                time.sleep(0.1)  # Brief pause on fitting errors
                
            except Exception as e:
                self.logger.error(f"Unexpected error in monitor loop: {str(e)}")
                time.sleep(1)

    def update(self, frame) -> Tuple[plt.Artist, ...]:
        """Update the visualization with new data"""
        try:
            # Update scatter plot
            new_positions: List[Tuple[float, float, float]] = []
            while not self.analyzer.centers_queue.empty():
                pos = self.analyzer.centers_queue.get()
                self.logger.debug(f"Position type: {type(pos)}, value: {pos}")
                if isinstance(pos, (tuple, list)) and len(pos) == 3:
                    new_positions.append(pos)
                else:
                    self.logger.error(f"Invalid data format: {pos}")

            if new_positions:
                self.logger.debug(f"Processing {len(new_positions)} positions")
                x_pos, y_pos, elapsed_time = map(np.array, zip(*new_positions))  # Convert to numpy arrays

                old_offsets = np.array(self.scatter.get_offsets())
                self.logger.debug(f"Old scatter data: {old_offsets}; Type: {type(old_offsets)}")

                # Ensure x_pos and y_pos are compatible with old_offsets
                new_offsets = np.vstack([old_offsets, np.column_stack([x_pos, y_pos])])
                self.logger.debug(f"New scatter data: {new_offsets}; Type: {type(new_offsets)}")
                self.scatter.set_offsets(new_offsets)

                # Auto-scale scatter plot with configurable padding
                padding = self.plot_padding
                self.center_plot.set_xlim(min(x_pos) - padding, max(x_pos) + padding)
                self.center_plot.set_ylim(min(y_pos) - padding, max(y_pos) + padding)

                # Retrieve old X and Y series data
                old_x_time, old_x_pos = map(np.array, self.line_x.get_data())
                self.logger.debug(f"Old X series data: {old_x_time, old_x_pos}; Type: {type(old_x_time)}, {type(old_x_pos)}")

                old_y_time, old_y_pos = map(np.array, self.line_y.get_data())
                self.logger.debug(f"Old Y series data: {old_y_time, old_y_pos}; Type: {type(old_y_time)}, {type(old_y_pos)}")

                # Update X series data
                if old_x_time.size == 0 or old_x_pos.size == 0:
                    self.line_x.set_data([elapsed_time], [x_pos])
                    self.logger.debug(f"New Updated X Time series: {self.line_x.get_data()}, Type: {type(self.line_x.get_data())}")
                else:
                    new_x_series_data = (np.concatenate([old_x_time, [elapsed_time]]), np.concatenate([old_x_pos, [x_pos]]))
                    self.line_x.set_data(new_x_series_data)
                    self.logger.debug(f"New Updated X Time series: {new_x_series_data}, Type: {type(new_x_series_data)}")

                # Update Y series data
                if old_y_time.size == 0 or old_y_pos.size == 0:
                    self.line_y.set_data([elapsed_time], [y_pos])
                    self.logger.debug(f"New Updated Y Time series: {self.line_y.get_data()}, Type: {type(self.line_y.get_data())}")
                else:
                    new_y_series_data = (np.concatenate([old_y_time, [elapsed_time]]), np.concatenate([old_y_pos, [y_pos]]))
                    self.line_y.set_data(new_y_series_data)
                    self.logger.debug(f"New Updated Y Time series: {new_y_series_data}, Type: {type(new_y_series_data)}")
                
                self.x_timeseries.relim()
                self.y_timeseries.relim()

                self.x_timeseries.autoscale_view()
                self.y_timeseries.autoscale_view()

        except Exception as e:
            self.logger.error(f"Error in visualization update: {e}")

            traceback.print_exc()  # This will print the full error traceback
            return self.scatter, self.line_x, self.line_y

    def _start_visualization(self):
        """Initialize and start the real-time visualization"""
        try:
            #Using GridSpec to customize layout
            #initialize figure
            self.main_dashboard = plt.figure(layout="constrained", animated=True)     

            self.gs = GridSpec(2,2, figure=self.main_dashboard)
            self.center_plot = self.main_dashboard.add_subplot(self.gs[:,0])
            self.x_timeseries = self.main_dashboard.add_subplot(self.gs[0,1])
            self.y_timeseries = self.main_dashboard.add_subplot(self.gs[1,1])
            self.y_timeseries.sharex(self.x_timeseries)      

            # Initialize scatter plot
            self.scatter = self.center_plot.scatter([], [], c='b', alpha=0.5)
            self.center_plot.set_xlabel('X Position (pixels)')
            self.center_plot.set_ylabel('Y Position (pixels)')
            self.center_plot.set_title('Beam Center Position')
            self.center_plot.grid(True)
            
            # Initialize time series
            self.line_x, = self.x_timeseries.plot([], [], 'Xb-', label='X Position')
            self.x_timeseries.set_xlabel('Time (s)')
            self.x_timeseries.set_ylabel('Position (pixels)')
            self.x_timeseries.set_title('Position vs Time')
            self.x_timeseries.grid(True)
            self.x_timeseries.legend()

            self.line_y, = self.y_timeseries.plot([], [], 'Xr-', label='Y Position')
            self.y_timeseries.set_xlabel('Time (s)')
            self.y_timeseries.set_ylabel('Position (pixels)')
            self.y_timeseries.set_title('Position vs Time')
            self.y_timeseries.grid(True)
            self.y_timeseries.legend()

            total_frames = self.total_points

            self.logger.info("Starting animation")
            self.ani = FuncAnimation(
                self.main_dashboard,
                self.update,
                interval=self.interval,
                blit=False,
                save_count=total_frames,
                cache_frame_data=False  # Try adding this
            )
            self.logger.info("Animation started")
            plt.show()
            
        except Exception as e:
            self.logger.error(f"Error starting visualization: {str(e)}")
            raise



            
        
            
        
    def stop(self):
        """Stop the monitoring system and save final data"""
        self.logger.info("Stopping beam monitor")
        self.running = False
        
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5.0)
            
        if self.cap:
            self.cap.release()
            
        cv2.destroyAllWindows()
        
        # Save final data and completion status
        self._save_final_data()
        self._save_completion_status()

    def _save_completion_status(self):
        """Save completion status and summary"""
        status_file = self.base_dir / "completion_status.txt"
        try:
            with open(status_file, 'w') as f:
                f.write(f"Monitoring completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Points collected: {self.points_collected}/{self.total_points}\n")
                f.write(f"Time logged: {self.logging_time} hours\n")
                
                # Add some basic statistics
                if len(self.positions_df) > 0:
                    f.write("\nBeam Position Statistics:\n")
                    f.write(f"X position mean: {self.positions_df['x'].mean():.2f}\n")
                    f.write(f"X position std: {self.positions_df['x'].std():.2f}\n")
                    f.write(f"Y position mean: {self.positions_df['y'].mean():.2f}\n")
                    f.write(f"Y position std: {self.positions_df['y'].std():.2f}\n")

            #save visualization plot
            final_plot = self.base_dir / "summary_plot.eps"
            self.main_dashboard.savefig(final_plot, transparent=True)

                    
        except Exception as e:
            self.logger.error(f"Error saving completion status: {e}")




logger = logging.getLogger(__name__)

def check_dependencies():
    """Check if all required dependencies are available"""
    try:
        import matplotlib
        logger.info(f"Matplotlib version: {matplotlib.__version__}")
        logger.info(f"Using backend: {matplotlib.get_backend()}")
        
        import numpy
        logger.info(f"NumPy version: {numpy.__version__}")
        
        import cv2
        logger.info(f"OpenCV version: {cv2.__version__}")
        
        import pandas
        logger.info(f"Pandas version: {pandas.__version__}")
        
        import scipy
        logger.info(f"SciPy version: {scipy.__version__}")
        
        return True
    except Exception as e:
        logger.error(f"Dependency check failed: {e}")
        return False

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Beam Monitoring System")
    parser.add_argument('-c', '--camera', type=int, help='Camera index to use')
    parser.add_argument('-i', '--interval', type=float, default=5.0, help='Interval in seconds for data logging and plotting')
    parser.add_argument('-t', '--time', type=float, default=0.5/60, help='Time to log data in hours')
    
    args = parser.parse_args()

    # Check dependencies first
    if not check_dependencies():
        print("Required dependencies are not properly installed.")
        print("Please check the error log for details.")
        return

    try:
        # Initialize camera selector
        selector = CameraSelector()
        
        camera_index = args.camera
        if camera_index is None:
            # If no camera index provided, prompt the user to select one
            while True:
                camera_index = selector.select_camera()
                if camera_index is None:
                    print("Camera selection cancelled")
                    return
                
                # Show preview and confirm selection
                if selector.preview_camera(camera_index):
                    break
                print("\nLet's try another camera...")
        
        print(f"\nStarting beam monitor with camera {camera_index}")
        
        # Initialize and start the beam monitor
        monitor = BeamMonitor(camera_index=camera_index, interval=args.interval, logging_time=args.time)
        monitor.start()
        
    except KeyboardInterrupt:
        print("\nStopping beam monitor...")
        if 'monitor' in locals():
            monitor.stop()
    except Exception as e:
        print(f"Error: {str(e)}")
        if 'monitor' in locals():
            monitor.stop()

if __name__ == "__main__":
    main()
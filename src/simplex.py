"""An animation using Simplex noise.


This module implements an animation of Simplex noise using py5 for Processing. 
The Display object is a wrapper around the RGBMatrix object from the (c) Henner 
Zeller h.zeller@acm.org LED-matrix library. At each frame, the Simplex noise
function is sampled and displayed in the Processing sketch. The numpy array
representation of the Processing sketch is then used to display the frame on
the LED matrices. 
"""

import sys

import py5
import pyaudio
import numpy as np
import cv2
from PIL import Image
from coloraide import Color

# from rgbmatrix import RGBMatrix, RGBMatrixOptions
from samplebase import SampleBase
from sound import AudioStream, AudioAnalyzer


# Dimensions for the LED matrix setup
LED_ROWS = 32
LED_COLS = 32
LED_CHAIN_LENGTH = 8
LED_PARALLEL = 2

# Parameters for the Processing sketch
SKETCH_WIDTH = LED_COLS * LED_CHAIN_LENGTH
SKETCH_HEIGHT = LED_ROWS * LED_PARALLEL
FRAME_RATE = 60
STEP_SIZE = 0.02

# Audio processing parameters
SAMPLE_RATE = 48000             # Hz
FORMAT = pyaudio.paFloat32      # 32-bit floating point
CHANNELS = 1                    # Mono
CHUNK_SIZE = SAMPLE_RATE // 60  # Synchronized for 60fps
FFT_SIZE = 1024                 # FFT size for spectral analysis


class ColorMap:
    """Represents a color palette for mapping points in a 2-D unit square
    to color coordinates."""

    def __init__(self, corners, size):
        """Initialize a ColorMap with the given corner colors and size.

        Corner colors are specified as a list of four colors in Oklch 
        color space. The input list should be ordered as follows:
            [top_left, top_right, bottom_left, bottom_right]
        
        Args:
            colors: A list of four colors, one for each corner of the unit
                square.
            size: A tuple (height, width) specifying the size of the color
                map.

        """
        self.top_left = Color('oklch', corners[0])
        self.top_right = Color('oklch', corners[1])
        self.bottom_left = Color('oklch', corners[2])
        self.bottom_right = Color('oklch', corners[3])

        # Initialize the color map
        self.cm = self.init_map(size[0], size[1])

    @staticmethod
    def default():
        """Create a default ColorMap with predefined corner colors."""
        corners = [
            [0.6102, 0.233, 292.61],
            [0.4496, 0.233, 292.61],
            [0.6102, 0.233, 323.65],
            [0.7443, 0.1785, 61.14],
        ] # defined using Oklch color space
        return ColorMap(corners, (SKETCH_HEIGHT, SKETCH_WIDTH))

    def init_map(self, height, width):
        """Initialize the color map for a given height and width."""
        colors = np.zeros((height, width, 3), dtype=np.uint8)

        # Interpolate along vertical edges
        left = Color.interpolate(
            [self.top_left, self.bottom_left],
            space='oklch',
            length=height
        )
        right = Color.interpolate(
            [self.top_right, self.bottom_right],
            space='oklch',
            length=height
        )

        # Interpolate along the horizontal axis for each row
        for y in range(height):
            y_norm = y / (height - 1)

            row = Color.interpolate(
                [left(y_norm), right(y_norm)],
                space='oklch',
                length=width
            )

            for x in range(width):
                x_norm = x / (width - 1)
                color = row(x_norm).convert('srgb').to_dict()
                r, g, b = [int(np.clip(c, 0, 1) * 255) for c in color['coords']]
                colors[y, x] = [r, g, b]

        return colors

    def get_color(self, y, x):
        """Get the color for a point in the color map."""
        return self.cm[y, x]

    def get_frame(self, y_indices, x_indices):
        """Get a frame of the color map."""
        return self.cm[y_indices, x_indices]
    

class NoiseGenerator:
    """Generates OpenSimplex noise for use in animation with ColorMap."""

    def __init__(self, height, width, range=2, step_size=0.05, recovery_rate=0.1, momentum_scale=0.005):
        self.xx, self.yy = np.meshgrid(
            np.linspace(0, range, num=width, dtype=np.float32),
            np.linspace(0, range, num=height, dtype=np.float32)
        )
        self.tt = np.zeros((1,), dtype=np.float32)
        self.time = 0.0
        self.default_step_size = step_size
        self.step_size = step_size
        self.velocity = 0.0  
        self.recovery_rate = recovery_rate  
        self.momentum_scale = momentum_scale  
    
    def get_frame(self):
        self.time += self.step_size
        self.tt[0] = self.time

        return py5.os_noise(
            self.xx, self.yy, self.tt.reshape(1,1,1)
        ).squeeze()


class Display(SampleBase):
    """A class representing the LED matrix setup.
    
    Attributes:
        matrix: an RGBMatrix object used to control the LEDs
        double_buffer: a FrameCanvas object, which is a buffer used to 
            read/write image frames to display on the LEDs
    """

    def __init__(self, *args, **kwargs):
        """Initializes the instance using the parameters in RGBMatrixOptions.
        
        Args:
            options: an RGBMatrixOptions object, which contains parameters
                that specify the LED matrix configuration
        """
        self.matrix = None
        super(Display, self).__init__(*args, **kwargs)
        self.process()
        self.double_buffer = self.matrix.CreateFrameCanvas()

    def update(self, pixels):
        """Write and display an image frame on the LEDs.
        
        Args:
            pixels (nd.array): an array representing the image frame
        """
        rgb_array = pixels[..., 1:]  # drop the A channel
        image = Image.fromarray(rgb_array, mode='RGB')

        self.double_buffer.SetImage(image, unsafe=False)
        self.double_buffer = self.matrix.SwapOnVSync(self.double_buffer)


# Configure LED matrix
# --led-rows=32 --led-cols=32 --led-slowdown-gpio=4 --led-no-hardware-pulse LED_NO_HARDWARE_PULSE --led-no-drop-privs
matrix = Display()

# Simplex noise parameters
t = 0.0
colormap = ColorMap.default()
noise_generator = NoiseGenerator(SKETCH_HEIGHT, 
                                 SKETCH_WIDTH, 
                                 step_size=STEP_SIZE)

# Audio
audio_stream = AudioStream(sample_rate=SAMPLE_RATE, 
                           chunk_size=CHUNK_SIZE, 
                           fft_size=FFT_SIZE, 
                           format=FORMAT, 
                           channels=CHANNELS,)
analyzer = AudioAnalyzer(n_bins=SKETCH_WIDTH, 
                         sample_rate=SAMPLE_RATE,
                         fft_size=FFT_SIZE,
                         min_db=-60,
                         max_db=30,)


def shift_channel(img, dx, dy):
    rows, cols = img.shape
    M = np.float32([[1, 0, dx], [0, 1, dy]])
    return cv2.warpAffine(img, M, (cols, rows), borderMode=cv2.BORDER_REFLECT)


def chromatic_aberration_cv2(img, shift_r=(2, 0), shift_g=(0, 0), shift_b=(-2, 0)):
    r = shift_channel(img[..., 0], *shift_r)
    g = shift_channel(img[..., 1], *shift_g)
    b = shift_channel(img[..., 2], *shift_b)
    return np.stack([r, g, b], axis=-1)


def create_frame(scale_factor=3, sigmaY=1):
    global t
    t += STEP_SIZE
    
    # Get audio data
    analyzer.update(audio_stream.buffer)
    energy = np.ones((SKETCH_HEIGHT, SKETCH_WIDTH)) * analyzer.smoothed_spectrum[np.newaxis, :]
    energy = np.power(energy, scale_factor)

    # Get noise data
    noise = noise_generator.get_frame()

    # Map noise and energy to color
    y_indices = py5.remap(noise, -1, 1, 0, SKETCH_HEIGHT - 1).astype(int)
    x_indices = py5.remap(energy, 0, 1, 0, SKETCH_WIDTH - 1).astype(int)
    out = colormap.get_frame(y_indices, x_indices)

    strength = analyzer.chrom_aberration_strength
    if strength > 0.01:
        print(f'{strength}')
        max_shift = 50  # max pixels of chromatic aberration
        shift_x = int(strength * max_shift)
        shift_y = int(strength * max_shift)

        out = chromatic_aberration_cv2(out, shift_r=(shift_x, shift_y), shift_b=(-shift_x, -shift_y))

    return out


def setup():
    """Processing setup function (called once)."""
    py5.size(SKETCH_WIDTH, SKETCH_HEIGHT)
    py5.frame_rate(FRAME_RATE)
    audio_stream.start()
    print("Running animation...")


def draw():
    """Processing draw function (called for each frame)."""
    # Update audio buffer
    audio_stream.update()

    # Update display
    out = create_frame()
    py5.set_np_pixels(out, bands='RGB')
    py5.load_np_pixels()
    matrix.update(py5.np_pixels)


if __name__ == '__main__':
    print(f'{np.__version__=}')
    
    # Run the animation
    try:
        print('Press CTRL-C to stop.')
        py5.run_sketch()
    except KeyboardInterrupt:
        sys.exit(0)

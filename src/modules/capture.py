"""A module for tracking useful in-game information."""

import time
import cv2
import threading
import ctypes
import mss
import mss.windows
import numpy as np
from src.common import config, utils
from ctypes import wintypes
user32 = ctypes.windll.user32
user32.SetProcessDPIAware()

# Get system DPI scaling factor
def get_dpi_scale():
    """Get the system DPI scaling factor."""
    try:
        # Get DPI for the primary monitor
        hdc = user32.GetDC(0)
        dpi = ctypes.windll.gdi32.GetDeviceCaps(hdc, 88)  # LOGPIXELSX
        user32.ReleaseDC(0, hdc)
        return dpi / 96.0  # 96 DPI is 100% scaling
    except:
        return 1.0


# The distance between the top of the minimap and the top of the screen
MINIMAP_TOP_BORDER = 5

# The thickness of the other three borders of the minimap
MINIMAP_BOTTOM_BORDER = 9

# Offset in pixels to adjust for windowed mode
WINDOWED_OFFSET_TOP = 36
WINDOWED_OFFSET_LEFT = 10

# The top-left and bottom-right corners of the minimap
MM_TL_TEMPLATE = cv2.imread('assets/minimap_tl_template.png', 0)
MM_BR_TEMPLATE = cv2.imread('assets/minimap_br_template.png', 0)

MMT_HEIGHT = max(MM_TL_TEMPLATE.shape[0], MM_BR_TEMPLATE.shape[0])
MMT_WIDTH = max(MM_TL_TEMPLATE.shape[1], MM_BR_TEMPLATE.shape[1])

# The player's symbol on the minimap
PLAYER_TEMPLATE = cv2.imread('assets/player_template.png', 0)
PT_HEIGHT, PT_WIDTH = PLAYER_TEMPLATE.shape


class Capture:
    """
    A class that tracks player position and various in-game events. It constantly updates
    the config module with information regarding these events. It also annotates and
    displays the minimap in a pop-up window.
    """

    def __init__(self):
        """Initializes this Capture object's main thread."""

        config.capture = self

        self.frame = None
        self.minimap = {}
        self.minimap_ratio = 1
        self.minimap_sample = None
        self.sct = None
        
        # Get DPI scaling and adjust default window size
        self.dpi_scale = get_dpi_scale()
        print(f'[~] Detected DPI scaling: {self.dpi_scale:.2f}x')
        
        # Start with a larger default window size for high-res displays
        self.window = {
            'left': 0,
            'top': 0,
            'width': 3440,  # Support ultrawide monitors
            'height': 1440
        }

        self.ready = False
        self.calibrated = False
        self.thread = threading.Thread(target=self._main)
        self.thread.daemon = True

    def start(self):
        """Starts this Capture's thread."""

        print('\n[~] Started video capture')
        self.thread.start()

    def _main(self):
        """Constantly monitors the player's position and in-game events."""

        mss.windows.CAPTUREBLT = 0
        while True:
            # Calibrate screen capture
            handle = user32.FindWindowW(None, 'MapleStory')
            rect = wintypes.RECT()
            user32.GetWindowRect(handle, ctypes.pointer(rect))
            rect = (rect.left, rect.top, rect.right, rect.bottom)
            rect = tuple(max(0, x) for x in rect)

            self.window['left'] = rect[0]
            self.window['top'] = rect[1]
            self.window['width'] = max(rect[2] - rect[0], MMT_WIDTH)
            self.window['height'] = max(rect[3] - rect[1], MMT_HEIGHT)

            # Calibrate by finding the top-left and bottom-right corners of the minimap
            with mss.mss() as self.sct:
                self.frame = self.screenshot()
            if self.frame is None:
                continue
                
            print(f'[~] Calibrating minimap detection on {self.window["width"]}x{self.window["height"]} window')
            print(f'[~] Template sizes - TL: {MM_TL_TEMPLATE.shape}, BR: {MM_BR_TEMPLATE.shape}, Player: {PLAYER_TEMPLATE.shape}')
            
            tl, _ = utils.single_match(self.frame, MM_TL_TEMPLATE)
            _, br = utils.single_match(self.frame, MM_BR_TEMPLATE)
            
            print(f'[~] Found minimap corners - TL: {tl}, BR: {br}')
            
            mm_tl = (
                tl[0] + MINIMAP_BOTTOM_BORDER,
                tl[1] + MINIMAP_TOP_BORDER
            )
            mm_br = (
                max(mm_tl[0] + PT_WIDTH, br[0] - MINIMAP_BOTTOM_BORDER),
                max(mm_tl[1] + PT_HEIGHT, br[1] - MINIMAP_BOTTOM_BORDER)
            )
            
            minimap_width = mm_br[0] - mm_tl[0]
            minimap_height = mm_br[1] - mm_tl[1]
            print(f'[~] Minimap dimensions: {minimap_width}x{minimap_height}')
            
            self.minimap_ratio = minimap_width / minimap_height
            print(f'[~] Minimap ratio: {self.minimap_ratio:.3f}')
            
            self.minimap_sample = self.frame[mm_tl[1]:mm_br[1], mm_tl[0]:mm_br[0]]
            self.calibrated = True
            print('[~] Minimap calibration complete')

            with mss.mss() as self.sct:
                while True:
                    if not self.calibrated:
                        break

                    # Take screenshot
                    self.frame = self.screenshot()
                    if self.frame is None:
                        continue

                    # Crop the frame to only show the minimap
                    minimap = self.frame[mm_tl[1]:mm_br[1], mm_tl[0]:mm_br[0]]

                    # Determine the player's position with adaptive threshold
                    # Lower threshold for high-resolution displays where scaling might affect matching
                    threshold = 0.6 if self.dpi_scale > 1.5 else 0.8
                    player = utils.multi_match(minimap, PLAYER_TEMPLATE, threshold=threshold)
                    if player:
                        config.player_pos = utils.convert_to_relative(player[0], minimap)
                        if len(player) > 1:
                            print(f'[~] Multiple player markers detected: {len(player)} (using first one)')
                    else:
                        # Try with even lower threshold as fallback
                        player = utils.multi_match(minimap, PLAYER_TEMPLATE, threshold=0.4)
                        if player:
                            config.player_pos = utils.convert_to_relative(player[0], minimap)
                            print(f'[~] Player detected with low confidence threshold')

                    # Package display information to be polled by GUI
                    self.minimap = {
                        'minimap': minimap,
                        'rune_active': config.bot.rune_active,
                        'rune_pos': config.bot.rune_pos,
                        'path': config.path,
                        'player_pos': config.player_pos
                    }

                    if not self.ready:
                        self.ready = True
                    time.sleep(0.001)

    def screenshot(self, delay=1):
        try:
            return np.array(self.sct.grab(self.window))
        except mss.exception.ScreenShotError:
            print(f'\n[!] Error while taking screenshot, retrying in {delay} second'
                  + ('s' if delay != 1 else ''))
            time.sleep(delay)

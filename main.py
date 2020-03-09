import cv2
import time
import pygame
import numpy as np

from djitellopy.tello import Tello
from ps3_inputs import ControllerEvents, PS3ControllerManager
from peeptree.processing import ImageProcessor

# Speed of the drone
# Frames per second of the pygame window display
S = 60
FPS = 25


class FrontEnd(object):

    """ 
    Maintains the Tello display and moves it through a PS3 controller.
    Press escape key to quit.
        The controls are:
            - Triangle button : Takeoff
            - X button : Land
            - Left joystick : Forward, backward, left and right.
            - Circle and Square buttons : Counter clockwise and clockwise rotations
            - Right joystick : Up and down.
    """

    def __init__(self):

        # Init pygame
        pygame.init()

        # Creat pygame window
        pygame.display.set_caption("Tello video stream")
        self.screen = pygame.display.set_mode([960, 720])

        # defining tello interaction object
        self.tello = Tello()

        # Drone velocities between -100~100
        self.for_back_velocity = 0
        self.left_right_velocity = 0
        self.up_down_velocity = 0
        self.yaw_velocity = 0
        self.speed = 10

        self.send_rc_control = False

        # creating a user event every 50 milliseconds (for updating)
        pygame.time.set_timer(pygame.USEREVENT + 1, 50)

    
    def run(self):

        ''' Tello control entry point '''

        # setting up the Tello for run
        if not self.tello.connect():
            print("Tello not connected")
            return
        if not self.tello.set_speed(self.speed):
            print("Not set speed to lowest possible")
            return

        # In case streaming is on. 
        # This happens when we quit this program without the escape key.
        if not self.tello.streamoff():
            print("Could not stop video stream")
            return
        if not self.tello.streamon():
            print("Could not start video stream")
            return

        crtl_manager = PS3ControllerManager()
        frame_read = self.tello.get_frame_read()
        processor = ImageProcessor("peeptree/classifier.pickle", block_size=20)

        should_stop = False
        while not should_stop:

            # handling controller events
            ctrl_event = crtl_manager.get_event()
            if ctrl_event in ControllerEvents.key_down_events:
                self.keydown(ctrl_event)
            elif ctrl_event in ControllerEvents.key_up_events:
                self.keyup(ctrl_event)

            # handling pygame  events
            for event in pygame.event.get():

                # sending velocities to Tello (update event) (every 50 ms)
                if event.type == pygame.USEREVENT + 1:
                    self.update()

                # handling escape command
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        should_stop = True

                # handling quit event (exit main control loop)
                elif event.type == pygame.QUIT:
                    should_stop = True                     

            # exit control when video streaming stops
            if frame_read.stopped:
                frame_read.stop()
                break

            if capture_counter == 5:

            # getting latest video frae from the drone
            # the "getter" for the frames is the call to "frame_read.frame"
            self.screen.fill([0, 0, 0])
            frame = cv2.cvtColor(frame_read.frame, cv2.COLOR_BGR2RGB)

            # processing the latest frame
            try:
                frame = processor.detect_object_segments(frame)
            except : print("error processing frame")
            
            # displaying the latest frame
            frame = np.rot90(frame)
            frame = np.flipud(frame)
            frame = pygame.surfarray.make_surface(frame)
            self.screen.blit(frame, (0, 0))
            pygame.display.update()

            # main control loop is limited by FPS
            time.sleep(1 / FPS)

        # deallocating control resources
        self.tello.end()


    def keydown(self, key):

        """ 
        Update velocities based on key pressed
        Arguments:
            key: pygame key
        """

        # set forward velocity
        # set backward velocity
        if key == ControllerEvents.JL_UP:  
            self.for_back_velocity = S
        elif key == ControllerEvents.JL_DOWN:  
            self.for_back_velocity = -S

        # set left velocity
        # set right velocity
        elif key == ControllerEvents.JL_LEFT:  
            self.left_right_velocity = -S
        elif key == ControllerEvents.JL_RIGHT:  
            self.left_right_velocity = S

        # set up velocity
        # set down velocity
        elif key == ControllerEvents.JR_UP:  
            self.up_down_velocity = S
        elif key == ControllerEvents.JR_DOWN:  
            self.up_down_velocity = -S

        # set yaw counter clockwise velocity
        # set yaw clockwise velocity
        elif key == ControllerEvents.CIRCLE_DOWN: 
            self.yaw_velocity = -S
        elif key == ControllerEvents.SQUARE_DOWN:  
            self.yaw_velocity = S


    def keyup(self, key):

        """ 
        Update velocities based on key released
        Arguments:
            key: pygame key
        """

        # set zero forward/backward velocity
        if key == ControllerEvents.JL_CENTER:  
            self.for_back_velocity = 0
            self.left_right_velocity = 0

        # set zero up/down velocity
        elif key == ControllerEvents.JR_CENTER:  
            self.up_down_velocity = 0

        # set zero yaw velocity
        elif key == ControllerEvents.CIRCLE_UP or key == ControllerEvents.SQUARE_UP:  
            self.yaw_velocity = 0

        # takeoff command
        elif key == ControllerEvents.TRIANGLE_UP:
            self.tello.takeoff()
            self.send_rc_control = True

        # land command
        elif key == ControllerEvents.X_UP:
            self.tello.land()
            self.send_rc_control = False


    def update(self):

        """ Update routine. Send velocities to Tello."""
        if self.send_rc_control:
            self.tello.send_rc_control(self.left_right_velocity, self.for_back_velocity, self.up_down_velocity,
                                       self.yaw_velocity)


def main():

    # running control entry point
    frontend = FrontEnd()
    frontend.run()


if __name__ == '__main__':
    main()
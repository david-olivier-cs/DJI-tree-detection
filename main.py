from djitellopy.tello import Tello
import cv2
import pygame
import numpy as np
import time

# Speed of the drone
# Frames per second of the pygame window display
S = 60
FPS = 25

class FrontEnd(object):

    """ 
    Maintains the Tello display and moves it through the keyboard keys.
        Press escape key to quit.
        The controls are:
            - T: Takeoff
            - L: Land
            - Arrow keys: Forward, backward, left and right.
            - A and D: Counter clockwise and clockwise rotations
            - W and S: Up and down.
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

        frame_read = self.tello.get_frame_read()

        # main control loop
        should_stop = False
        while not should_stop:

            # handling user events
            for event in pygame.event.get():

                # sending velocities to Tello (update evemt) (every 50 ms)
                if event.type == pygame.USEREVENT + 1:
                    self.update()

                # handling movement commands from user
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        should_stop = True
                    else:
                        self.keydown(event.key)

                # commands stop uppon key release
                elif event.type == pygame.KEYUP:
                    self.keyup(event.key)

                # handling quit event (exit main control loop)
                elif event.type == pygame.QUIT:
                    should_stop = True

            # exit control when video streaming stops
            if frame_read.stopped:
                frame_read.stop()
                break

            # displying the latest frame recevied by the drone
            # the "getter" for the frames is the call to "frame_read.frame"
            self.screen.fill([0, 0, 0])
            frame = cv2.cvtColor(frame_read.frame, cv2.COLOR_BGR2RGB)
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

        if key == pygame.K_UP:  # set forward velocity
            self.for_back_velocity = S
        elif key == pygame.K_DOWN:  # set backward velocity
            self.for_back_velocity = -S

        elif key == pygame.K_LEFT:  # set left velocity
            self.left_right_velocity = -S
        elif key == pygame.K_RIGHT:  # set right velocity
            self.left_right_velocity = S

        elif key == pygame.K_w:  # set up velocity
            self.up_down_velocity = S
        elif key == pygame.K_s:  # set down velocity
            self.up_down_velocity = -S

        elif key == pygame.K_a:  # set yaw counter clockwise velocity
            self.yaw_velocity = -S
        elif key == pygame.K_d:  # set yaw clockwise velocity
            self.yaw_velocity = S


    def keyup(self, key):

        """ 
        Update velocities based on key released
        Arguments:
            key: pygame key
        """

        if key == pygame.K_UP or key == pygame.K_DOWN:  # set zero forward/backward velocity
            self.for_back_velocity = 0

        elif key == pygame.K_LEFT or key == pygame.K_RIGHT:  # set zero left/right velocity
            self.left_right_velocity = 0

        elif key == pygame.K_w or key == pygame.K_s:  # set zero up/down velocity
            self.up_down_velocity = 0

        elif key == pygame.K_a or key == pygame.K_d:  # set zero yaw velocity
            self.yaw_velocity = 0

        # takeoff command
        elif key == pygame.K_t:  # takeoff
            self.tello.takeoff()
            self.send_rc_control = True

        # land command
        elif key == pygame.K_l:  # land
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
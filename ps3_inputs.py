''' This module enables PS3 input detection '''

import time
import redis
from inputs import get_gamepad
from multiprocessing import Process

class ControllerEvents():

    ''' Defines codes for controller states '''

    NO_EVENT = 99

    # left joystick states
    JL_UP = 0
    JL_DOWN = 1
    JL_LEFT = 2
    JL_RIGHT = 3
    JL_CENTER = 4
    # right joystick states
    JR_UP = 5
    JR_DOWN = 6
    JR_CENTER = 7
    # button states
    SQUARE_DOWN = 8
    SQUARE_UP = 9
    CIRCLE_DOWN = 10
    CIRCLE_UP = 11
    X_DOWN = 12
    X_UP = 13
    TRIANGLE_DOWN = 14
    TRIANGLE_UP = 15

    # set to identify "key up" events
    key_up_events = set([X_UP, SQUARE_UP, TRIANGLE_UP, CIRCLE_UP, JL_CENTER, JR_CENTER])
    
    # set to identify "key down" events
    key_down_events = set([X_DOWN, SQUARE_DOWN, TRIANGLE_DOWN, CIRCLE_DOWN, 
                          JL_DOWN, JL_UP, JL_LEFT, JL_RIGHT, JR_UP, JR_DOWN])

class DBModel():

    ''' Interface for accessing redis '''

    # defining default db parameters
    connection_pool = None
    redis_config =  {"host" : "localhost", "port" : 6379, "decode_responses" : True}


    def __init__(self):

        # first instance creates a connection pool
        if self.connection_pool is None:
            DBModel.connection_pool = redis.ConnectionPool(**self.redis_config)
            self.r_server = redis.StrictRedis(connection_pool=self.connection_pool)
            self.init_redis_vars()
        else:
            self.r_server = redis.StrictRedis(connection_pool=self.connection_pool)


    def init_redis_vars(self):

        ''' Setting the initial state of redis vars '''

        self.r_server.set("input_detection_active", "0")
        while self.get_input_event() is not None: pass
      

    def start_detection(self):
        self.r_server.set("input_detection_active", "1")
    def stop_detection(self):
        self.r_server.set("input_detection_active", "0")
    def check_detection(self):
        return self.r_server.get("input_detection_active") == "1"


    def add_input_event(self, request_str):
        self.r_server.lpush("input_events", request_str)
    def get_input_event(self):
        return self.r_server.rpop("input_events")


class PS3ControllerManager():

    ''' Detects and manages inputs from a connected PS3 controller '''

    def __init__(self):
    
        self.db_model = DBModel()
        self.input_detection_h = None

        # launching input detection as seoerate process
        self.launch_input_detection()


    def launch_input_detection(self):

        ''' Launches the detection process and waits '''

        self.db_model.start_detection()
        self.input_detection_h = Process(target=self.detect_target_inputs)
        self.input_detection_h.start()


    def stop_input_detection(self):
        
        ''' Stops the detection process '''

        self.db_model.stop_detection()
        self.input_detection_h.join()


    def detect_target_inputs(self):

        ''' Detecting specific controller events '''

        try:

            # defining container fir detected events
            detected_events = []

            # setting initial controller input states
            jr_state = ControllerEvents.JR_CENTER
            jl_state = ControllerEvents.JL_CENTER
            x_button_state = ControllerEvents.X_UP
            square_btn_state = ControllerEvents.SQUARE_UP
            circle_btn_state = ControllerEvents.CIRCLE_UP
            triangle_btn_state = ControllerEvents.TRIANGLE_UP

            # run untill detection is halted
            while(self.db_model.check_detection()):

                # going through broadcasted events
                crtl_events = get_gamepad()
                for crtl_event in crtl_events:

                    # left joystick - left to right motion
                    if crtl_event.code == "ABS_X":
                        
                        if (not jl_state == ControllerEvents.JL_UP) and (not jl_state == ControllerEvents.JL_DOWN): 

                            if crtl_event.state >= 100 and crtl_event.state <= 170 and not (jl_state == ControllerEvents.JL_CENTER):
                                jl_state = ControllerEvents.JL_CENTER
                                detected_events.append(jl_state)
                            
                            elif crtl_event.state < 100 and not (jl_state == ControllerEvents.JL_LEFT):
                                jl_state = ControllerEvents.JL_LEFT
                                detected_events.append(jl_state)
                                
                            elif crtl_event.state > 170 and not (jl_state == ControllerEvents.JL_RIGHT):
                                jl_state = ControllerEvents.JL_RIGHT
                                detected_events.append(jl_state)

                    # left joystick - up and down motion
                    if crtl_event.code == "ABS_Y":
                        
                        if (not jl_state == ControllerEvents.JL_LEFT) and (not jl_state == ControllerEvents.JL_RIGHT):

                            if crtl_event.state >= 100 and crtl_event.state <= 170 and not (jl_state == ControllerEvents.JL_CENTER):
                                jl_state = ControllerEvents.JL_CENTER
                                detected_events.append(jl_state)

                            elif crtl_event.state < 100 and not (jl_state == ControllerEvents.JL_UP):
                                jl_state = ControllerEvents.JL_UP
                                detected_events.append(jl_state)
                                
                            elif crtl_event.state > 170 and not (jl_state == ControllerEvents.JL_DOWN):
                                jl_state = ControllerEvents.JL_DOWN
                                detected_events.append(jl_state)

                    # right joystick - up and down motion
                    elif crtl_event.code == "ABS_RY":

                        if crtl_event.state >= 100 and crtl_event.state <= 170 and not (jr_state == ControllerEvents.JR_CENTER):
                                jr_state = ControllerEvents.JR_CENTER
                                detected_events.append(jr_state)

                        elif crtl_event.state < 100 and not (jr_state == ControllerEvents.JR_UP):
                            jr_state = ControllerEvents.JR_UP
                            detected_events.append(jr_state)
                            
                        elif crtl_event.state > 170 and not (jr_state == ControllerEvents.JR_DOWN):
                            jr_state = ControllerEvents.JR_DOWN
                            detected_events.append(jr_state)

                    # x button press
                    elif crtl_event.code == "BTN_THUMBR" :

                        if crtl_event.state == 1 and x_button_state == ControllerEvents.X_UP:
                            x_button_state = ControllerEvents.X_DOWN
                            detected_events.append(x_button_state)

                        elif crtl_event.state == 0 and x_button_state == ControllerEvents.X_DOWN:
                            x_button_state = ControllerEvents.X_UP
                            detected_events.append(x_button_state)

                    # square button press
                    elif crtl_event.code == "BTN_START" :

                        if crtl_event.state == 1 and square_btn_state == ControllerEvents.SQUARE_UP:
                            square_btn_state = ControllerEvents.SQUARE_DOWN
                            detected_events.append(square_btn_state)

                        elif crtl_event.state == 0 and square_btn_state == ControllerEvents.SQUARE_DOWN:
                            square_btn_state = ControllerEvents.SQUARE_UP
                            detected_events.append(square_btn_state)

                    # circle button press
                    elif crtl_event.code == "BTN_THUMBL" :

                        if crtl_event.state == 1 and circle_btn_state == ControllerEvents.CIRCLE_UP:
                            circle_btn_state = ControllerEvents.CIRCLE_DOWN
                            detected_events.append(circle_btn_state)

                        elif crtl_event.state == 0 and circle_btn_state == ControllerEvents.CIRCLE_DOWN:
                            circle_btn_state = ControllerEvents.CIRCLE_UP
                            detected_events.append(circle_btn_state)

                    # trianglebutton press
                    elif crtl_event.code == "BTN_SELECT" :

                        if crtl_event.state == 1 and triangle_btn_state == ControllerEvents.TRIANGLE_UP:
                            triangle_btn_state = ControllerEvents.TRIANGLE_DOWN
                            detected_events.append(triangle_btn_state)

                        elif crtl_event.state == 0 and triangle_btn_state == ControllerEvents.TRIANGLE_DOWN:
                            triangle_btn_state = ControllerEvents.TRIANGLE_UP
                            detected_events.append(triangle_btn_state)

                # adding detected events to the event queue in redis
                for detected_event in detected_events:
                    self.db_model.add_input_event(str(detected_event))
                detected_events.clear()

                time.sleep(0.005)

        # dont stop on failure
        except : pass


    def get_event(self):

        ''' 
        Returns the next event from the event queue 
        
        Returns
        -------
        (int) : (ControllerEvents) event code
        '''
        
        event = self.db_model.get_input_event()
        
        if event is not None: 
            event = int(event)
        else: 
            event = ControllerEvents.NO_EVENT
        
        return event


if __name__ == "__main__":

    ''' Running a demo when the script is launched as a stand alone '''

    crtl_manager = PS3ControllerManager()

    try:


        while(crtl_manager.db_model.check_detection()):

            # getting the latest event
            event = crtl_manager.get_event()

            if event is not None:

                # left joystick events
                if event == ControllerEvents.JL_CENTER:
                    print("Joystick left - center")
                elif event == ControllerEvents.JL_LEFT:
                    print("Joystick left - left")
                elif event == ControllerEvents.JL_RIGHT:
                    print("Joystick left - right")
                elif event == ControllerEvents.JL_UP:
                    print("Joystick left - up")
                elif event == ControllerEvents.JL_DOWN:
                    print("Joystick left - down")

                # rignt joystick events
                if event == ControllerEvents.JR_CENTER:
                    print("Joystick right - center")
                elif event == ControllerEvents.JR_UP:
                    print("Joystick right - up")
                elif event == ControllerEvents.JR_DOWN:
                    print("Joystick right - down")

                # button events
                elif event == ControllerEvents.SQUARE_DOWN:
                    print("Square button down")
                elif event == ControllerEvents.SQUARE_UP:
                    print("Square button up")
                elif event == ControllerEvents.CIRCLE_DOWN:
                    print("Circle button down")
                elif event == ControllerEvents.CIRCLE_UP:
                    print("Circle button up")
                elif event == ControllerEvents.X_DOWN:
                    print("X button down")
                elif event == ControllerEvents.X_UP:
                    print("X button up")
                elif event == ControllerEvents.TRIANGLE_DOWN:
                    print("Triangle button down")
                elif event == ControllerEvents.TRIANGLE_UP:
                    print("Triangle button up")

            time.sleep(0.005)

    except:
        print("Controller input detection halted")
        controller_manager.stop_input_detection()
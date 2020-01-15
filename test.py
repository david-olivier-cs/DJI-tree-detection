from djitellopy.tello import Tello

def main():

    ''' Note that thw time between commands was changed from 1 sec to 0.5 sec '''

    # creating drone connectivity object
    tello = Tello()

    # defining
    try:

        # starting choreography
        tello.connect()
        tello.takeoff()

        # performing movements
        for _ in range(5):
            tello.move_down(30)
            tello.move_up(30)

        # ending choreography
        tello.land()
        tello.end()

    except:

        # making sure the drone land
        tello.land()
        tello.end()


if __name__ == "__main__":
    main()
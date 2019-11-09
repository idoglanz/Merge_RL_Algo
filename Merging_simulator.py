import matplotlib.pyplot as plt
from http.server import HTTPServer, BaseHTTPRequestHandler
from io import BytesIO
import time as time
import csv
import random
from Merge_algorithm import *
import sys

NUM_ROWS = 50
NUM_LINES = 5
DISTANCE_BETWEEN_STUDS = 10
MIN_CAR_MASS = 1000
MAX_CAR_MASS = 4000
MIN_TRUCK_MASS = 4000
MAX_TRUCK_MASS = 8000
CSV_HEADER = ["ts", "road", "id", "type", "lane", "x", "y", "speed", "acc", "dvdt"]
sample_rate = 30  # in Hz (refresh rate of browser)
first_ever = True
packet_id = 0
ROAD_LENGTH = 820
LANES = 4   # including merge
LANE_WIDTH = 4
SCALE = 2
ROAD_LIMITS = [0, 0, ROAD_LENGTH*SCALE, LANES*LANE_WIDTH*SCALE]
MAX_SPEED = 120/3.6
MERGE_BOUND = ROAD_LENGTH / 4
CAR_WIDTH = int(3 * SCALE)   # after multiplying with the scale factor

N_VEHICLES = 1
EPISODE_LENGTH = 256
COUNTER = 0
TOTAL_RUNS = 0
INIT_SPEED = 120
MAX_SAMPLES_PER_EPISODE = 512

PREV_STATE = np.zeros((ROAD_LENGTH*SCALE, LANES * LANE_WIDTH * SCALE))
PREV_ACTION = 'NOTHING'
PREV_REWARD = 0


class Road:
    def __init__(self, length, lanes, resolution):
        self.road_length = length
        self.n_lanes = lanes
        self.resolution = resolution
        self.merge_position = self.road_length/2
        self.lane_width = LANE_WIDTH
        self.free_space = np.zeros((self.road_length*resolution, self.n_lanes * self.lane_width * resolution))

    def draw_road(self, car):
        frame = self.free_space + 3*car.current_position
        # plt.imshow(self.free_space.transpose(), aspect=5, cmap='hot', origin='lower', interpolation='none')
        plt.imshow(frame.transpose(), aspect=5, cmap='hot', origin='lower', interpolation='none')
        plt.show(False)
        plt.pause(0.00005)
        plt.clf()

    def reset_road(self):
        self.free_space = np.zeros((self.road_length*self.resolution, self.n_lanes * self.lane_width * self.resolution))


class Vehicles:
    def __init__(self):
        self.vehicles = {}

    def update(self, id, ts, x, y):
        if id in self.vehicles:
            if len(self.vehicles[id]["x"]) > 0 and x < self.vehicles[id]["x"][-1]:

                # Update cycle and sequence ID
                self.vehicles[id]["cycle"] += 1
                self.vehicles[id]["sequence_id"] = 1000 * self.vehicles[id]["cycle"] + id

    def get_vehicle_seq_id(self, id):
        return self.vehicles[id]["sequence_id"]

    def get_vehicle_mass_and_length(self, id, type):
        if id in self.vehicles:
            return self.vehicles[id]["mass"], self.vehicles[id]["length"]
        if type == "car":
            mass = random.uniform(MIN_CAR_MASS, MAX_CAR_MASS)
            length = random.uniform(3, 5.5)

        else:
            mass = random.uniform(MIN_TRUCK_MASS, MAX_TRUCK_MASS)
            length = random.uniform(5.5, 15)

        self.vehicles[id] = {"mass": mass, 'length': length, "t": [], "x": [], "y": [], "cycle": 0, "sequence_id": id}
        return mass, length


class MergingVehicle:
    def __init__(self, length, road_layout):
        self.length = length
        self.speed = 0
        self.x = 0
        self.y = 0.5
        self.in_frame = True
        self.current_position = road_layout
        self.status = 'Fine'   # indicate the status - collision, out-of-road, finished segment

    def set_speed(self, speed):
        self.speed = speed / 3.6

    def move(self, direction):
        speed_scale = 1.1
        if direction is None:
            return True
        else:
            if self.x >= ROAD_LENGTH - self.length and self.y >= 1.5:
                self.status = 'finished'  # car is trying to go oob
                return False
            elif self.x >= MERGE_BOUND - self.length and self.y <= 1.5:
                self.status = 'failed_to_merge'
                return False
            elif self.y < 0 or self.y >= LANES - 0.5:
                self.in_frame = False
                self.status = 'illegal_action'  # car is trying to go oob
                return False
            elif direction is 'SLOW':
                self.speed /= speed_scale
                self.x += self.speed * (1/sample_rate) * SCALE
                return True
            elif direction is 'NOTHING':
                self.x += self.speed * (1/sample_rate) * SCALE
                return True
            elif direction is 'FAST':
                self.speed = self.speed*speed_scale if self.speed*speed_scale <= MAX_SPEED else self.speed
                self.x += self.speed * (1/sample_rate) * SCALE
                return True
            elif direction is 'LEFT_SLOW':
                self.speed /= speed_scale
                self.x += self.speed * (1/sample_rate) * SCALE
                self.y += 0.1 * self.speed/10
                return True
            elif direction is 'LEFT':
                self.x += self.speed * (1/sample_rate) * SCALE
                self.y += 0.1 * self.speed/10
                return True
            elif direction is 'LEFT_FAST':
                self.speed = self.speed*speed_scale if self.speed*speed_scale <= MAX_SPEED else self.speed
                self.x += self.speed * (1/sample_rate) * SCALE
                self.y += 0.1 * self.speed/10
                return True
            elif direction is 'RIGHT_SLOW':
                self.speed /= speed_scale
                self.x += self.speed * (1/sample_rate) * SCALE
                self.y -= 0.1 * self.speed/10
                return True
            elif direction is 'RIGHT':
                self.x += self.speed * (1/sample_rate) * SCALE
                self.y -= 0.1 * self.speed/10
                return True
            elif direction is 'RIGHT_FAST':
                self.speed = self.speed*speed_scale if self.speed*speed_scale <= MAX_SPEED else self.speed
                self.x += self.speed * (1/sample_rate) * SCALE
                self.y -= 0.1 * self.speed/10
                return True
            else:
                print('illegal Action!')

    def speed_control(self, action):
        if action is 'acc':
            self.speed += 1
        elif action is 'de-acc':
            self.speed -= 1
        else:
            print('Not a valid action')

    def check_collision(self, free_space):
        # Check if collided with other vehicles
        if np.sum(np.multiply(free_space, self.current_position)) == 0:
            return False
        else:
            return True

    def reset(self):
        self.speed = INIT_SPEED / 3.6
        self.x = 0
        self.y = 0.5
        self.in_frame = True
        self.current_position = np.zeros_like(self.current_position)
        self.status = 'Fine'   # indicate the status - collision, out-of-road, finished segment

    def reset_free_space(self):
        self.current_position = np.zeros_like(self.current_position)


def float_conv(s):
    try:
        s = float(s)
    except ValueError:
        pass
    return s


class SimulationClock:
    def __init__(self):
        self.CLOCK = int(round(time.time() * 1000))
        self.prev_dt = 0
        self.update_flag = True
        # self.sync_dt = VS.sync_dt
        self.last_sync = self.CLOCK
        self.TOLERANCE = 3

    def update_clock(self, dt):
        self.CLOCK += (dt - self.prev_dt) * 1000
        self.prev_dt = dt
        # print('Simulation time:', self.CLOCK)

    def reset_flag(self):
        self.update_flag = False

    def set_flag(self):
        self.update_flag = True

    # def time_for_packet(self):
    #     # print(abs(self.CLOCK - self.last_sync))
    #     if abs(self.CLOCK - self.last_sync) >= self.sync_dt:
    #         print('Time to send a packet to server!')
    #         self.last_sync = self.CLOCK
    #         return True
    #     else:
    #         return False


class SimpleHTTPRequestHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        return

    def do_POST(self):
        global packet_id, clock, first_ever, new_car, COUNTER
        global TOTAL_RUNS, PREV_ACTION, PREV_REWARD, PREV_STATE

        content_length = int(self.headers['Content-Length'])
        body = self.rfile.read(content_length).decode("utf-8")
        reader = csv.reader(body.split('\n'), delimiter=',')

        # free_space = road.free_space
        res = road.resolution

        for row in reader:
            if len(row) > 2:
                d = {CSV_HEADER[i]: float_conv(row[i]) for i in range(len(CSV_HEADER))}
                # omit traffic cones, lights etc.
                if d['id'] < 200:
                    continue

                # d["lane"] += 1
                d["lane"] = round(d['y']) + 1
                d['y'] += 1.5
                d["mass"], d['length'] = vehicles.get_vehicle_mass_and_length(d["id"], d["type"])

                if clock.update_flag:
                    if first_ever:  # runs only on first iteration to sync with simulation time
                        clock.prev_dt = d['ts']
                        clock.update_clock(d['ts'])
                        # system.update_clocks(clock.CLOCK)
                        first_ever = False
                    else:
                        clock.update_clock(d['ts'])  # keep track of clock (based on simulation clock)
                    clock.reset_flag()

                d['ts'] = clock.CLOCK

                vehicles.update(d["id"], d["ts"], d["x"], d["y"])

                # seq_id = vehicles.get_vehicle_seq_id(d['id'])

                # Draw vehicle on road

                road.free_space[round(res*d['x']):round(res*d['x']) + int(d['length']*res), round(res*d['y'])*road.lane_width - int(CAR_WIDTH/2):round(res*d['y'])*road.lane_width + int(CAR_WIDTH/2)] = d['speed']

                # print('Stud ID:', d['id'], ', Position (X/Y):', d['x'], '/', d['y'], ', Lane:', d['lane'], d['length'])

        # Flip lanes (just because we are not in England)

        road.free_space[:, LANE_WIDTH*SCALE:] = np.fliplr(road.free_space[:, LANE_WIDTH*SCALE:])

        # -------------------------------- THIS IS WHERE MERGING VEHICLES ARE ADDED ------------------------------------

        if COUNTER <= MAX_SAMPLES_PER_EPISODE and mode == 'Learn':

            for new_vehicle in range(N_VEHICLES):   # iterate over new vehicles

                if new_car.in_frame is True:

                    # plot merging vehicle in free space
                    new_car.reset_free_space()

                    new_car.current_position[round(res * new_car.x):round(res * new_car.x) + int(new_car.length * res), round(res * new_car.y) * road.lane_width - int(CAR_WIDTH/2):round(res * new_car.y) * road.lane_width + int(CAR_WIDTH/2)] = new_car.speed

                    road.free_space += new_car.current_position

                    # Calculate action (forward pass):
                    epsilon = 0.5 - TOTAL_RUNS * 0.01   # should be replaced with a more sophisticated decay method

                    action = get_action(fix_dim(road.free_space), model=model, ep=epsilon if epsilon > 0 else 0)

                    print('Sample #:', COUNTER, 'Action taken:', action, ', speed:', new_car.speed, 'Prev Reward:', PREV_REWARD)

                    # Execute (if possible) and get action feasibility feedback

                    is_action_taken = new_car.move(action)

                    # we now test whether action is feasible in terms of road boundaries only
                    if is_action_taken is False:
                        if new_car.status is "finished":
                            reward = 1000
                            new_car.reset()
                            data_set.n_games += 1
                            print('Finished successfully')
                        if new_car.status is 'failed_to_merge':
                            reward = -400
                            new_car.reset()
                            data_set.n_games += 1
                            print('Failed to merge')
                        elif new_car.status is "illegal_action":
                            reward = -400
                            new_car.reset()
                            print('tried to perform illegal action')
                        else:
                            reward = 0
                            print('not supposed to be in this state')

                        data_set.append(state=road.free_space, action=action, reward=reward, next_state=road.free_space)
                        new_car.reset()

                    else:
                        # test to see if collision occurs with other vehicles
                        if new_car.check_collision(np.round(road.free_space - new_car.current_position)) is True:
                            reward = -300
                            data_set.append(state=road.free_space, action=action, reward=reward, next_state=road.free_space)
                            print('Collision Detected, resetting')
                            new_car.status = 'collision'
                            data_set.n_games += 1
                            new_car.reset()
                        else:

                            data_set.append(state=PREV_STATE,
                                            action=PREV_ACTION,
                                            reward=PREV_REWARD,
                                            next_state=road.free_space)

                            PREV_STATE = road.free_space
                            PREV_ACTION = action
                            PREV_REWARD = calc_reward(new_car)

            COUNTER += 1

        elif mode == 'Run':

            for new_vehicle in range(N_VEHICLES):   # iterate over new vehicles

                if new_car.in_frame is True:

                    # plot merging vehicle in free space
                    new_car.reset_free_space()

                    new_car.current_position[round(res * new_car.x):round(res * new_car.x) + int(new_car.length * res), round(res * new_car.y) * road.lane_width - int(CAR_WIDTH/2):round(res * new_car.y) * road.lane_width + int(CAR_WIDTH/2)] = new_car.speed

                    road.free_space += new_car.current_position

                    # Calculate action (forward pass):
                    epsilon = 0.5 - TOTAL_RUNS * 0.01   # should be replaced with a more sophisticated decay method
                    action = get_action(fix_dim(road.free_space), model=model, ep=epsilon if epsilon > 0 else 0)
                    print('Sample #:', COUNTER, 'Action taken:', action, ', speed:', new_car.speed)

                    # Execute (if possible) and get action feasibility feedback

                    is_action_taken = new_car.move(action)

                    # we now test whether action is feasible in terms of road boundaries only
                    if is_action_taken is False:
                        if new_car.status is "finished":
                            reward = 100
                            new_car.reset()
                            data_set.n_games += 1
                            print('Finished successfully')
                        if new_car.status is 'failed_to_merge':
                            reward = -100
                            new_car.reset()
                            data_set.n_games += 1
                            print('Failed to merge')
                        elif new_car.status is "illegal_action":
                            reward = -100
                            new_car.reset()
                            print('tried to perform illegal action')
                        else:
                            reward = 0
                            print('not supposed to be in this state')

                        new_car.reset()

                    else:
                        # test to see if collision occurs with other vehicles
                        if new_car.check_collision(np.round(road.free_space - new_car.current_position)) is True:
                            print('Collision Detected, resetting')
                            new_car.status = 'collision'
                            data_set.n_games += 1
                            new_car.reset()

            COUNTER += 1

        else:
            # PERFORM LEARNING CYCLE
            TOTAL_RUNS += 1
            print('Starting learning cycle #', TOTAL_RUNS)
            data_set.cut_first_sample()   # just because its rubbish
            train(model, optimizer, loss_fn, data_set, 5)   # train the model with the current batch
            data_set.clear()       # clear the replay memory
            new_car.reset()        # Reset the game (just the merging car)
            COUNTER = 0

        if mode == 'Run':
            road.draw_road(new_car)

        road.draw_road(new_car)    # for plotting only

        road.reset_road()   # resets the road matrix for new frame (from simulator)

        clock.set_flag()    # Sets clock flag for n

        self.send_response(200)
        self.send_header("Content-type", "text/plain")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        response = BytesIO()
        response.write(b'This is POST request. ')
        response.write(b'Received: ')
        self.wfile.write(response.getvalue())


if __name__ == '__main__':

    try:
        mode = sys.argv[1]

    except:
        print('Please specify "Learn" or "Run" to indicate running mode, default will be Learn')
        mode = 'Learn'

    print('Mode:', mode)

    clock = SimulationClock()

    print('Start time:', clock.CLOCK)

    road = Road(length=ROAD_LENGTH, lanes=LANES, resolution=SCALE)

    print('Road Length:', ROAD_LENGTH, ', Lanes:', LANES, 'Resolution:', SCALE)

    vehicles = Vehicles()

    new_car = MergingVehicle(5, np.zeros_like(road.free_space))

    new_car.set_speed(INIT_SPEED)

    data_set = DataSet(EPISODE_LENGTH, road.free_space)

    if mode == 'Learn':
        # model, optimizer, loss_fn = create_model(road.free_space.shape)
        model, optimizer, loss_fn = create_model(np.expand_dims(road.free_space, 2).shape)

    elif mode == 'Run':
        model = tf.keras.models.load_model('merge_simulator_net.h5')

    # model, optimizer, loss_fn = create_model(np.expand_dims(road.free_space, 2).shape)

    httpd = HTTPServer(('localhost', 8000), SimpleHTTPRequestHandler)

    httpd.serve_forever()

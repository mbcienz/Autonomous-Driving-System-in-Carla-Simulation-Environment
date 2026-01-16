# Copyright (c) # Copyright (c) 2018-2020 CVC.
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.


""" This module implements an agent that roams around a track following random
waypoints and avoiding other vehicles. The agent also responds to traffic lights,
traffic signs, and has different possible configurations. """

import random
import numpy as np
import carla
from basic_agent import BasicAgent
from local_planner import RoadOption
from behavior_types import Cautious, Aggressive, Normal, Customized

from misc import get_speed, positive, is_within_distance, compute_distance

import math
from shapely.geometry import Polygon
from perception import Perception

# Constants
TRAFFIC_LIGHT_FILTER = "*traffic_light*"
VEHICLE_FILTER = "*vehicle*"
STATIC_PROP_FILTER = "*static.prop*"
BIKE_TYPES = ['vehicle.gazelle.omafiets', 'vehicle.bh.crossbike', 'vehicle.diamondback.century']

class BehaviorAgent(BasicAgent):
    """
    BehaviorAgent implements an agent that navigates scenes to reach a given
    target destination, by computing the shortest possible path to it.
    This agent can correctly follow traffic signs, speed limitations,
    traffic lights, while also taking into account nearby vehicles. Lane changing
    decisions can be taken by analyzing the surrounding environment such as tailgating avoidance.
    Adding to these are possible behaviors, the agent can also keep safety distance
    from a car in front of it by tracking the instantaneous time to collision
    and keeping it in a certain range. Finally, different sets of behaviors
    are encoded in the agent, from cautious to a more aggressive ones.
    """

    def __init__(self, vehicle, behavior='customized', opt_dict={}, map_inst=None, grp_inst=None):
        """
        Constructor method.
        """
        super().__init__(vehicle, opt_dict=opt_dict, map_inst=map_inst, grp_inst=grp_inst)
        self._look_ahead_steps = 0

        # Vehicle information
        self._speed = 0
        self._speed_limit = 0
        self._direction = RoadOption.LANEFOLLOW
        self._min_speed = 5
        self._behavior = None
        self._is_overtaking = False
        self._overtake_list = []
        self._lateral_safety_margin = 2.5
        self._max_lateral_steer = 0.5
        self._stop_sign_counter = 0
        self._stop_id=0
        self._stop=0
        self._stop_counter = 60
        self._walker_stop_time = {}

        # Parameters for agent behavior
        if behavior == 'cautious':
            self._behavior = Cautious()

        elif behavior == 'normal':
            self._behavior = Normal()

        elif behavior == 'aggressive':
            self._behavior = Aggressive()

        elif behavior == 'customized':
            self._behavior = Customized()
        
        # Perception system
        self.Perception = Perception(self._behavior, self._world, self._vehicle, self._local_planner)


    #################### UTILITY METHODS - START ####################     

    def _set_speed(self, speed):
        """
        This method sets the speed of the vehicle, taking into account the speed limit.
        """
        new_speed = min(speed, self._speed_limit - self._behavior.speed_lim_dist , self._behavior.max_speed)
        if new_speed>=0:
            self._local_planner.set_speed(new_speed)
        else:
            print("ERROR: negative speed!")

    def _update_information(self):
        """
        This method updates the information regarding the ego vehicle based on the surrounding world.
        """
        self._speed = get_speed(self._vehicle)
        self._speed_limit = self._vehicle.get_speed_limit()
        self._set_speed(self._speed_limit)
        self._direction = self._local_planner.target_road_option
        if self._direction is None:
            self._direction = RoadOption.LANEFOLLOW

        self._look_ahead_steps = int((self._speed_limit) / 10)

        self._incoming_waypoint, self._incoming_direction = self._local_planner.get_incoming_waypoint_and_direction(
            steps=self._look_ahead_steps)
        if self._incoming_direction is None:
            self._incoming_direction = RoadOption.LANEFOLLOW

    def _tailgating(self, waypoint, vehicle_list):
        """
        This method is in charge of tailgating behaviors.
        """

        left_turn = waypoint.left_lane_marking.lane_change
        right_turn = waypoint.right_lane_marking.lane_change

        left_wpt = waypoint.get_left_lane()
        right_wpt = waypoint.get_right_lane()

        behind_vehicle_state, behind_vehicle, _ = self._vehicle_obstacle_detected(vehicle_list, max(
            self._behavior.min_proximity_threshold, self._speed_limit / 2), up_angle_th=180, low_angle_th=160)
        if behind_vehicle_state and self._speed < get_speed(behind_vehicle):
            if (right_turn == carla.LaneChange.Right or right_turn ==
                    carla.LaneChange.Both) and waypoint.lane_id * right_wpt.lane_id > 0 and right_wpt.lane_type == carla.LaneType.Driving:
                new_vehicle_state, _, _ = self._vehicle_obstacle_detected(vehicle_list, max(
                    self._behavior.min_proximity_threshold, self._speed_limit / 2), up_angle_th=180, lane_offset=1)
                if not new_vehicle_state:
                    print("Tailgating, moving to the right!")
                    end_waypoint = self._local_planner.target_waypoint
                    self._behavior.tailgate_counter = 200
                    self.set_destination(end_waypoint.transform.location,
                                         right_wpt.transform.location)
            elif left_turn == carla.LaneChange.Left and waypoint.lane_id * left_wpt.lane_id > 0 and left_wpt.lane_type == carla.LaneType.Driving:
                new_vehicle_state, _, _ = self._vehicle_obstacle_detected(vehicle_list, max(
                    self._behavior.min_proximity_threshold, self._speed_limit / 2), up_angle_th=180, lane_offset=-1)
                if not new_vehicle_state:
                    print("Tailgating, moving to the left!")
                    end_waypoint = self._local_planner.target_waypoint
                    self._behavior.tailgate_counter = 200
                    self.set_destination(end_waypoint.transform.location,
                                         left_wpt.transform.location)

    def car_following_manager(self, vehicle_ahead, distance: float, debug: bool = False):
        """
        Module in charge of car-following behaviors when there's someone in front of us.
        """
        control = None
        vehicle_ahead_speed = get_speed(vehicle_ahead)
        delta_v = max(1, (self._speed - vehicle_ahead_speed) / 3.6)
        ttc = distance / delta_v if delta_v != 0 else distance / np.nextafter(0., 1.)

        # Under safety time distance, slow down
        if self._behavior.safety_time > ttc > 0.0:
            self._set_speed(min(int(vehicle_ahead_speed * 2 / 3), self._behavior.speed_decrease))
            control = self._local_planner.run_step(debug=debug)

        # Actual safety distance area, try to follow the speed of the vehicle in front
        elif 2 * self._behavior.safety_time > ttc >= self._behavior.safety_time:
            self._set_speed(vehicle_ahead_speed)
            control = self._local_planner.run_step(debug=debug)

        # Normal behavior
        else:
            self._set_speed(self._behavior.max_speed)
            control = self._local_planner.run_step(debug=debug)

        return control
    
    def emergency_stop(self):
        """
        Overwrites the throttle a brake values of a control to perform an emergency stop.
        The steering is kept the same to avoid going out of the lane when stopping during turns.
        """
        control = carla.VehicleControl()
        control.throttle = 0.0
        control.brake = self._max_brake
        control.hand_brake = False
        return control

    def _is_a_bike(self, vehicle) -> bool:
        '''
        Check if the vehicle is a bike.
        '''
        return vehicle.type_id in BIKE_TYPES

    def _same_lane(self, target_vehicle, target_vehicle_transform, ego_wpt):
        '''
        This function checks if two vehicles are in the same lane.
        '''
        target_vehicle_bb = target_vehicle.bounding_box.get_world_vertices(target_vehicle_transform)
        is_in_same_lane_id = False
        for point_location in target_vehicle_bb:
            if (self._map.get_waypoint(point_location, lane_type=carla.LaneType.Any).lane_id == (ego_wpt.lane_id)):
                is_in_same_lane_id = True
                break
        return is_in_same_lane_id
    
    def normal_driving(self, debug=False):
        """
        This function defines the standard behavior of the vehicle, in the absence of critical situations.
        """
        if debug:
            print("Normal behavior. \n")
        self._set_speed(self._behavior.max_speed) # set the speed to the maximum allowed
        control = self._local_planner.run_step(debug=debug)
        return control

    #################### UTILITY METHODS - END ####################


    #################### PEDESTRIAN HANDLER - START ####################

    def pedestrian_handler(self, ego_wp, debug=False):
        """
        Function that manages braking in the presence of pedestrians.
        """
        # Pedestrian perception
        pedestrian_state, pedestrian, p_distance = self.Perception.pedestrian_perception(ego_wp, self._direction)
        
        if pedestrian_state:
            if debug:
                print("Pedestrian detected!")

            # p_distance is computed from the center of pedestrian and vehicle, we use bounding boxes to calculate the actual distance
            distance = p_distance - max(
                pedestrian.bounding_box.extent.y, pedestrian.bounding_box.extent.x) - max(
                    self._vehicle.bounding_box.extent.y, self._vehicle.bounding_box.extent.x)
            
            # Emergency brake if the car is very close
            if distance < self._behavior.braking_distance:
                if debug:
                    print("Emergency stop for pedestrian!")
                return self.emergency_stop()
        return None
    
    #################### PEDESTRIAN HANDLER - END ####################


    #################### STOP SIGN HANDLER - START ####################

    def stop_sign_handler(self, debug=False):
        """
        Function that manages stopping at a stop sign and checks if it's safe to proceed.
        """
        stop_state, stop_temp= self.Perception.stop_sign_perception()
        if stop_state and (stop_temp.id != self._stop_id): 
            if self._stop < self._stop_counter and self._speed != 0.0:
                if debug:
                    print("Stopping at STOP sign!")
                self._stop += 1 
                return self.emergency_stop()
            else:
                if debug:
                    print("Proceeding")
                self._stop=0
                self._stop_id = stop_temp.id
                speed = min([self._behavior.max_speed, self._speed_limit - self._behavior.speed_lim_dist]) 
                self._set_speed(speed)
                control = self._local_planner.run_step(debug=False)
                return control
        return None
    
    #################### STOP SIGN HANDLER - END ####################


    #################### TRAFFIC LIGHT HANDLER - START ####################

    def traffic_light_handler(self):
        """
        This method is in charge of behaviors for red lights.
        """
        actor_list = self._world.get_actors()
        lights_list = actor_list.filter("*traffic_light*")
        affected, _ = self._affected_by_traffic_light(lights_list)

        return affected
    
    #################### TRAFFIC LIGHT HANDLER - END ####################


    #################### JUNCTION HANDLER - START ####################

    def junction_handler(self, ego_vehicle_loc, my_wp, curve_wp, debug=False):
        '''
        This function manages the vehicle's behavior at road junction.
        '''
        # Check if the vehicle is in a junction
        if self.Perception.get_next_waypoint().is_junction:
            if debug:
                print("Junction!")

            # Checks for the presence of a stop sign
            control = self.stop_sign_handler(debug=True)
            if control is not None:
                return control
            
            # Check if the vehicle must give precedence to other vehicles
            if self.junction_perception(ego_vehicle_loc, my_wp, debug=debug):
                if debug:
                    print("Stopping for precedence!")
                return self.emergency_stop()
            
            # Slow down in curve
            if self._is_curve(my_wp, curve_wp):
                self._set_speed(10)
            
            control = self._local_planner.run_step(debug=debug)
            return control
        
        return None
    
    def _is_curve(self, wp1, wp2, threshold_degrees=10):
        '''
        Checks if the change in orientation between two waypoints indicates a curve.
        '''
        dir1 = wp1.transform.get_forward_vector()
        dir2 = wp2.transform.get_forward_vector()
        
        # Calculate angle between the two vectors
        dot_product = dir1.x * dir2.x + dir1.y * dir2.y + dir1.z * dir2.z
        dot_product = max(-1.0, min(1.0, dot_product))
        angle = math.degrees(math.acos(dot_product))
        
        return angle > threshold_degrees

    def junction_perception(self, ego_vehicle_loc, my_wp, max_distance=10, debug=False):
        """
        This function checks if the vehicle must give precedence to other vehicles at a junction.
        """
        actors = self._world.get_actors().filter('*vehicle*')
        
        for other in actors:
            if other.id == self._vehicle.id:
                continue # for checking only other vehicles
    
            other_loc = other.get_location()
            distance = ego_vehicle_loc.distance(other_loc)
            other_speed = get_speed(other)

            if distance > max_distance:
                continue # for checking only nearby vehicles
            
            # 1. Stationary vehicle ahead in the same lane
            same_lane = self._same_lane(other, other.get_transform(), my_wp)
            if same_lane and distance < 5 and other_speed < 1.0:
                if debug:
                    print("Stationary vehicle ahead in the same lane!")
                return True
            
            # 2. Stationary vehicle ahead in target lane
            next_wpt = self._local_planner.get_incoming_waypoint_and_direction(steps=3)[0]
            if next_wpt:
                if self._same_lane(other, other.get_transform(), next_wpt) and distance < 5 and other_speed < 1.0:
                    if debug:
                        print("Stationary vehicle ahead in target lane!")
                    return True
            
            # 3. Vehicle crossing my path
            if self._vehicle_crossing_my_path(ego_vehicle_loc, other, my_wp, max_distance, debug):
                if debug:
                    print("Vehicle crossing my path!")
                return True
            
            # 4. Moving vehicle too close
            if distance < 5 and other_speed > 1.0:
                if debug:
                    print("Moving vehicle too close!")
                return True
        
        return False

    def _vehicle_crossing_my_path(self, ego_location, other_vehicle, my_wp, max_distance, debug=False):
        """
        This function checks if a vehicle is crossing my trajectory (using polygons).
        """
        try:
            # Checks minimum speed of the other vehicle
            other_speed = get_speed(other_vehicle)/3.6
            if other_speed < 1.0:  # vehicle too slow
                return False
            
            # Build the polygon of my future trajectory
            route_bb = []
            ego_transform = self._vehicle.get_transform()
            extent_y = self._vehicle.bounding_box.extent.y
            
            # Add my starting point
            r_vec = ego_transform.get_right_vector()
            p1 = ego_location + carla.Location(extent_y * r_vec.x, extent_y * r_vec.y)
            p2 = ego_location + carla.Location(-extent_y * r_vec.x, -extent_y * r_vec.y)
            route_bb.append([p1.x, p1.y])
            route_bb.append([p2.x, p2.y])
            
            # Add my future waypoints to the trajectory
            current_wp = my_wp
            waypoint_count = 0
            max_waypoints = 10
            while current_wp and waypoint_count < max_waypoints:
                if ego_location.distance(current_wp.transform.location) > max_distance:
                    break
                r_vec = current_wp.transform.get_right_vector()
                p1 = current_wp.transform.location + carla.Location(extent_y * r_vec.x, extent_y * r_vec.y)
                p2 = current_wp.transform.location + carla.Location(-extent_y * r_vec.x, -extent_y * r_vec.y)
                route_bb.append([p1.x, p1.y])
                route_bb.append([p2.x, p2.y])
                next_wps = current_wp.next(2.0)
                if next_wps:
                    current_wp = next_wps[0]
                    waypoint_count += 1
                else:
                    break
            
            # Checks if there are enough points to create a polygon
            if len(route_bb) < 3:
                return False
            
            # Create the polygon of my trajectory
            ego_polygon = Polygon(route_bb)
            
            # Checks if the other vehicle is too far away
            other_location = other_vehicle.get_location()
            if ego_location.distance(other_location) > max_distance:
                return False
            
            # Builds the other vehicle's polygon
            other_bb_points = []
            other_transform = other_vehicle.get_transform()
            other_bb = other_vehicle.bounding_box
            
            # Add the other vehicle's current position
            other_vertices = other_bb.get_world_vertices(other_transform)
            for v in other_vertices:
                other_bb_points.append([v.x, v.y])
            
            # Estimate the other vehicle's future trajectory
            other_forward = other_transform.get_forward_vector()
            future_distance = other_speed * self._behavior.safety_time
            
            # Add the other vehicle's future waypoints to the trajectory
            for i in range(1, 6):
                future_location = other_location + carla.Location(
                    other_forward.x * (future_distance * i / 5),
                    other_forward.y * (future_distance * i / 5),
                    0
                )
                future_transform = carla.Transform(future_location, other_transform.rotation)
                future_vertices = other_bb.get_world_vertices(future_transform)
                for v in future_vertices:
                    other_bb_points.append([v.x, v.y])
            
            # Checks if there are enough points to create a polygon
            if len(other_bb_points) < 3:
                return False
            
            # Create the other vehicle's polygon
            other_polygon = Polygon(other_bb_points)
            
            # Check for intersection between polygons
            if ego_polygon.intersects(other_polygon):
                ego_speed = self._speed/3.6
                if ego_speed > 1.0:
                    intersection = ego_polygon.intersection(other_polygon)
                    print("Intersection between polygons!")
                    if hasattr(intersection, 'centroid'):
                        intersection_point = intersection.centroid
                        intersection_loc = carla.Location(intersection_point.x, intersection_point.y, 0)
                        my_time = ego_location.distance(intersection_loc) / ego_speed
                        other_time = other_location.distance(intersection_loc) / other_speed
                        if other_time <= my_time + self._behavior.safety_time/2:
                            if debug:
                                print("Vehicle crossing my path, precedence to other vehicle!")
                            return True
                else:
                    if debug:
                        print("My vehicle is stationary, precedence to other vehicle!")
                    return True
                
            return False
        
        # In case of error, precedence to other vehicle for safety
        except Exception as e:
            if debug:
                print(f"Error {e} - precedence to other vehicle for safety!")
            return True
        
    #################### JUNCTION HANDLER - END ####################


    #################### VEHICLE HANDLER - START ####################

    def vehicle_handler(self, ego_vehicle_loc, ego_vehicle_wp, debug=False):
        '''
        This function defines the behavior for vehicles ahead.
        '''
        # Vehicle perception
        vehicle_list = self.Perception.vehicle_perception(ego_vehicle_wp)
        
        # Check the presence of vehicles for the lane change
        if self._direction == RoadOption.CHANGELANELEFT:
            vehicle_ahead_state, vehicle_ahead, distance_ahead = self._vehicle_obstacle_detected(
                vehicle_list, max(
                    self._behavior.min_proximity_threshold, self._speed_limit / 2), up_angle_th=180, lane_offset=-1) 
        elif self._direction == RoadOption.CHANGELANERIGHT:
            vehicle_ahead_state, vehicle_ahead, distance_ahead = self._vehicle_obstacle_detected(
                vehicle_list, max(
                    self._behavior.min_proximity_threshold, self._speed_limit / 2), up_angle_th=180, lane_offset=1)
        # Check the presence of vehicles for juction
        elif ego_vehicle_wp.is_junction:
            vehicle_ahead_state, vehicle_ahead, distance_ahead = self._vehicle_obstacle_detected(
                vehicle_list, max(
                    self._behavior.min_proximity_threshold, self._speed_limit / 3), up_angle_th=90)
        # Check the presence of vehicles for lane following
        else:
            vehicle_ahead_state, vehicle_ahead, distance_ahead = self._vehicle_obstacle_detected(
                vehicle_list, max(
                    self._behavior.min_proximity_threshold, self._speed_limit / 3), up_angle_th=60)            

            # Check for tailgating
            if not vehicle_ahead_state and self._direction == RoadOption.LANEFOLLOW \
                    and not ego_vehicle_wp.is_junction and self._speed > 10 \
                    and self._behavior.tailgate_counter == 0:
                self._tailgating(ego_vehicle_wp, vehicle_list)

        # If there is a vehicle ahead that is not a bike
        if vehicle_ahead_state and not self._is_a_bike(vehicle_ahead):
            # distance_ahead is computed from the center of the two vehicles, we use bounding boxes to calculate the actual distance
            distance = distance_ahead - max(
                vehicle_ahead.bounding_box.extent.y, vehicle_ahead.bounding_box.extent.x) - max(
                    self._vehicle.bounding_box.extent.y, self._vehicle.bounding_box.extent.x) 
            
            # Emergency brake if the car is very close
            if distance < self._behavior.braking_distance:
                if debug:
                    print("Emergency stop for vehicle ahead!")
                return self.emergency_stop()
            else:
                # Follow the vehicle ahead
                return self.car_following_manager(vehicle_ahead, distance)
        
        # Manages the behavior of the vehicle when the opposite vehicle invade my lane
        return self.steer_for_other_vehicle_in_my_lane(ego_vehicle_loc, ego_vehicle_wp, control=None, debug=debug)
    
    def steer_for_other_vehicle_in_my_lane(self, ego_vehicle_loc, ego_vehicle_wp, control=None, debug=False):
        """
        This function manages the behavior of the vehicle when the opposite vehicle invade my lane.
        """
        # If the vehicle is at a junction, do not apply corrections
        if not ego_vehicle_wp.is_junction:
            # Vehicle perception
            nearby_vehicles = self.Perception.vehicle_perception(ego_vehicle_wp)
            
            for vehicle in nearby_vehicles:
                if not self._is_a_bike(vehicle):
                    vehicle_loc = vehicle.get_location()
                    vehicle_wp = self._map.get_waypoint(vehicle_loc)
                    
                    # Check if the vehicle is in the opposite lane (lane offset=2)
                    opposite_lane_offset = -1 if ego_vehicle_wp.lane_id > 0 else 1
                    if vehicle_wp.lane_id == ego_vehicle_wp.lane_id + 2 * opposite_lane_offset:
                        ego_transform = self._vehicle.get_transform()
                        
                        # Compute the vector between the two vehicles
                        delta_loc = vehicle_loc - ego_vehicle_loc
                        
                        # Get forward (longitudinal) and right (lateral) vectors and normalize them
                        forward_vec = ego_transform.get_forward_vector()
                        forward_vec = np.array([forward_vec.x, forward_vec.y, forward_vec.z])
                        forward_vec = forward_vec / np.linalg.norm(forward_vec)
                        right_vec = ego_transform.get_right_vector()
                        right_vec = np.array([right_vec.x, right_vec.y, right_vec.z])
                        right_vec = right_vec / np.linalg.norm(right_vec)
                        
                        # Project delta_loc onto forward and right vectors
                        delta_loc_array = np.array([delta_loc.x, delta_loc.y, delta_loc.z])
                        longitudinal_dist = np.dot(delta_loc_array, forward_vec)
                        lateral_dist = np.dot(delta_loc_array, right_vec)

                        # If lateral distance is less than margin and vehicle is close and ahead of me
                        if abs(lateral_dist) < self._lateral_safety_margin and longitudinal_dist < 12 and longitudinal_dist > 0:
                            if debug:
                                print("The opposite vehicle is close and invading my lane!")
                            
                            # Calculate correction (move right if vehicle is on the left)
                            correction_direction = 1.0
                            correction_factor = min(0.3, (self._lateral_safety_margin - abs(lateral_dist)) / self._lateral_safety_margin)
                            steer_correction = 0.3 * correction_factor * correction_direction
                            
                            # Apply the correction
                            if control is None:
                                control = self._local_planner.run_step()
                            control.steer = np.clip(control.steer + steer_correction, -self._max_lateral_steer, self._max_lateral_steer)
                            break
        return control
    
    #################### VEHICLE HANDLER - END ####################


    #################### BIKE HANDLER - START ####################

    def bike_handler(self, ego_wp, debug=False):
        '''
        This function defines the behavior for bike ahead.
        '''
        # Bike perception
        bike_state, bike, b_distance = self.Perception.bike_perception(ego_wp, self._direction)

        if bike_state:
            if debug:
                print("Bike detected!")

            # b_distance is computed from the center of the two vehicles, we use bounding boxes to calculate the actual distance
            distance = b_distance - max(
            bike.bounding_box.extent.y, bike.bounding_box.extent.x) - max(
                self._vehicle.bounding_box.extent.y, self._vehicle.bounding_box.extent.x)
            
            # If the vehicle is close to the bike and there are no cars in the overtaking lane, overtake the bike with steer
            if distance < 3 and not self.Perception.check_for_vehicle_on_overtaking_lane(ego_wp, 15)[0]:
                print("Overtake the bike with steer!")
                control = carla.VehicleControl()
                control.throttle = 1.0
                control.steer = self.last_steer - 0.1 # left steer
                return control
            else:
                # Emergency brake if the bike is very close
                if distance < 1:
                    if debug:
                        print("Emergency stop for bike!")
                    return self.emergency_stop()
                # Follow the bike
                if distance < 4:
                    return self.car_following_manager(bike, distance)

        return None
    
    #################### BIKE HANDLER - END ####################


    #################### OBSTACLE HANDLER - START ####################

    def obstacle_handler(self, ego_vehicle_wp, debug=False):
        """
        This function manages the behaviour of ego vehicle in presence of obstacles (vehicle obstacle and static obstacles)
        """

        # Vehicle perception        
        vehicle_list = self.Perception.vehicle_perception(ego_vehicle_wp, distance=60)
        vehicle_state, vehicle, distance = self._vehicle_obstacle_detected(vehicle_list, max(
                    self._behavior.min_proximity_threshold, self._speed_limit / 2), up_angle_th=90, lane_offset=1)

        # Static object perception
        static_list = self.Perception.static_object_perception(ego_vehicle_wp)
        static_state, static_obs, static_distance = self.Perception.ahead_obstacle_detected(static_list, max(
                    self._behavior.min_proximity_threshold, self._speed_limit / 2), up_angle_th=90, lane_offset=0)

        # Cone perception
        cone_list = self.Perception.cone_perception(ego_vehicle_wp)

        # a. VEHICLE OBSTACLE HANDLING

        # If there is a vehicle (obstacle) ahead that is not a bike on my same lane
        if vehicle_state and not self._is_a_bike(vehicle) and self._same_lane(vehicle, vehicle.get_transform(), ego_vehicle_wp):
        
            # distance is computed from the center of the two vehicles, we use bounding boxes to calculate the actual distance
            distance = distance - max(
                vehicle.bounding_box.extent.y, vehicle.bounding_box.extent.x) - max(
                    self._vehicle.bounding_box.extent.y, self._vehicle.bounding_box.extent.x)
            
            # Get obstacle information
            obstacle_l = self.Perception.obstacle_perception(vehicle_list,lane_offset=1)
            obstacle_length = self._compute_obstacle_length(obstacle_l)
            if obstacle_length == 0:
                return self._local_planner.run_step(debug=debug)

            # Reduce speed in the presence of obstacles
            speed = max (1,(get_speed(self._vehicle) * distance) / 50)
            
            # Check if the obstacle is within braking distance
            if distance < self._behavior.braking_distance and obstacle_l:
                # Check for incoming vehicles on the overtaking lane
                obstacle_state, obstacle_vehicle, obstacle_distance = self.Perception.check_for_vehicle_on_overtaking_lane(ego_vehicle_wp, (distance+obstacle_length+25)*2)

                # Compute total overtaking distance
                overtaking_distance = obstacle_length + distance
                speed = min([self._behavior.max_speed, self._speed_limit - self._behavior.speed_lim_dist]) / 3.6
                time_to_overtake = overtaking_distance / speed

                # If the tail of the obstacle is further than the estimated position (distance + length)
                if (distance + obstacle_length) - obstacle_l[-1][2] < 0:
                    security_offset = 7 # add a big security offset
                else:
                    security_offset = 3 # add a small security offset

                # If oncoming vehicle detected, check if overtaking is safe
                if obstacle_state:
                    return self.evaluate_safety_time_overtake(obstacle_vehicle, obstacle_distance, time_to_overtake, overtaking_distance, distance, speed, security_offset)
                else:
                    # Perform overtaking maneuver
                    self._local_planner.set_speed(speed * 3.6)
                    print("Overtaking vehicle ahead!")
                    return self.overtake(overtaking_distance + security_offset)
            
            # If the vehicle is far and stationary, proceed with local planner
            elif get_speed(vehicle) < 0.3:
                self._local_planner.set_speed(speed * 3.6)
                return self._local_planner.run_step(debug=debug)
                
            # If the vehicle is far and moving, follow it
            else:
                return self.car_following_manager(vehicle, distance, side=True)

        # b. STATIC OBSTACLE HANDLING

        # If there is a static obstacle
        elif static_state:
            
            # Get obstacle information
            obstacle_l= self.Perception.obstacle_perception(static_list,lane_offset=0)
            obstacle_length = self._compute_obstacle_length(obstacle_l)
            if obstacle_length == 0:
                return self._local_planner.run_step(debug=debug)
            
            # Check if the first obstacle is within a safe braking distance
            if obstacle_l[0][2] < self._behavior.braking_distance + 10 and obstacle_l:
                # Check for oncoming vehicles on the overtaking lane
                obstacle_state, obstacle_vehicle, obstacle_distance = self.Perception.check_for_vehicle_on_overtaking_lane(ego_vehicle_wp, (static_distance+obstacle_length+20)*(self._speed_limit / self._behavior.max_speed ))
                
                # Compute total overtaking distance
                overtaking_distance = obstacle_length + obstacle_l[0][2] + 10
                speed = min([self._behavior.max_speed, self._speed_limit - self._behavior.speed_lim_dist]) / 3.6
                time_to_overtake = overtaking_distance / speed

                # If oncoming vehicle detected, check if overtaking is safe
                if obstacle_state:
                    return self.evaluate_safety_time_overtake(obstacle_vehicle, obstacle_distance, time_to_overtake, overtaking_distance, distance, speed )
                else:
                    # Update list to track handled obstacles
                    for o in obstacle_l:
                        if o not in self._overtake_list:
                            self._overtake_list.append(o[1].id)
                    if len(cone_list)>0:
                        for cone in cone_list:
                            if cone not in self._overtake_list:
                                self._overtake_list.append(o[1].id)
                    
                    # Perform overtaking maneuver
                    self._local_planner.set_speed(speed * 3.6)
                    print("Overtaking static obstacle ahead!")
                    return self.overtake(overtaking_distance - 2)
            
            # If the obstacle is far, proceed with local planner
            else:
                speed = min([self._behavior.max_speed, self._speed_limit - self._behavior.speed_lim_dist]) / 3.6
                self._local_planner.set_speed(speed * 3.6)
                return self._local_planner.run_step(debug=debug)

    def _compute_obstacle_length(self, obstacle_l):
        '''
        This function computes the length of the obstacle based on the list of obstacles detected.
        '''
        if len(obstacle_l) == 1:
            sv = obstacle_l[0]
            obstacle_length = max(sv[1].bounding_box.extent.x, sv[1].bounding_box.extent.y) * 2
        elif len(obstacle_l) >1:
            obstacle_length = obstacle_l[-1][2] - obstacle_l[0][2]
        else:
            obstacle_length = 0
        return obstacle_length

    def evaluate_safety_time_overtake(self, obstacle_vehicle, obstacle_distance, time_to_overtake, overtaking_distance, distance, speed, security_offset = 0):
        '''
        This function evaluates if the ego vehicle has enough space and time to complete an overtaking manoeuvre with respect to an obstacle present in the lane.
        '''

        # Speed of the obastcle vehicle
        danger_vehicle_speed = get_speed(obstacle_vehicle) / 3.6
        
        # Compute where the obstacle vehicle will be after the overtake time
        s = (obstacle_distance - danger_vehicle_speed * (time_to_overtake + 3))
        
        # Check if there is enough space to overtake safely
        if s > overtaking_distance + security_offset: 
            # If the ego vehicle is close to the obstacle (closer than braking distance) start overtaking 
            if distance < self._behavior.braking_distance:
                self._local_planner.set_speed(speed * 3.6)
                control = self.overtake(overtaking_distance + security_offset)
                print("Overtake!")
            # Otherwise, keep following the lane until closer
            else:
                control = self._local_planner.run_step(debug=True)
        else:    
            # Not enough space to overtake, emergency stop
            return self.emergency_stop()
        return control

    def overtake(self, distance_for_overtaking, direction='left', two_way=True):
        """
        This function generates the overtaking path and performs the overtaking maneuver.
        """
        
        wp = self._map.get_waypoint(self._vehicle.get_location())
        step_distance = self._sampling_resolution
        
        # Save the current global route to reconnect later
        old_plan = self._local_planner._waypoints_queue

        # Initialize the overtaking plan with current lane-follow
        plan = []
        plan.append((wp, RoadOption.LANEFOLLOW))

        # Move forward to prepare the lane change
        next_wp = wp.next(step_distance)[0]

        # Determine the side lane based on overtaking direction
        if direction == 'left':
            side_wp = next_wp.get_left_lane()
            plan.append((next_wp, RoadOption.CHANGELANELEFT))
            plan.append((side_wp, RoadOption.LANEFOLLOW))
        else:
            side_wp = next_wp.get_right_lane()
            plan.append((next_wp, RoadOption.CHANGELANERIGHT))
            plan.append((side_wp, RoadOption.LANEFOLLOW))

        # Move along the overtaking lane for the specified distance
        distance = 0
        while distance < distance_for_overtaking:
            if two_way:
                next_wps = plan[-1][0].previous(step_distance) # for two-way roads
            else:
                next_wps = plan[-1][0].next(step_distance) # for one-way roads
            next_wp = next_wps[0]

            # Increment the traveled distance on the overtaking lane
            distance += next_wp.transform.location.distance(plan[-1][0].transform.location)
            
            # Add the waypoint to continue lane following
            plan.append((next_wp, RoadOption.LANEFOLLOW))

        # Prepare to return to the original lane
        next_wp = plan[-1][0].next(step_distance)[0]
        
        # Return to the right lane
        side_wp = next_wp.get_right_lane()
        if side_wp is not None:
            plan.append((next_wp, RoadOption.LANEFOLLOW))
            plan.append((side_wp, RoadOption.CHANGELANERIGHT))
        else:
            print("No right lane available — skipping lane change.")

        # Reconnect to the original global plan after the overtaking
        old_plan_wp = list(map(lambda x: x[0], old_plan))
        start_index = self._global_planner._find_closest_in_list(plan[-1][0], old_plan_wp)
        for i in range(start_index, len(old_plan_wp)):
            plan.append(self._local_planner._waypoints_queue[i])

        # Apply the new global plan with overtaking
        self.set_global_plan(plan)

        # Execute the next control command based on the new plan
        return self._local_planner.run_step(debug=False)

    #################### OBSTACLE HANDLER - END ####################


    #################### MAIN LOOP - START ####################

    def run_step(self, debug=True):
        """
        Execute one step of navigation.

            :param debug: boolean for debugging
            :return control: carla.VehicleControl
        """
        self._update_information()
        self.Perception._update_information(self._speed_limit, self._speed, self._look_ahead_steps)
        
        control = None
        if self._behavior.tailgate_counter > 0:
            self._behavior.tailgate_counter -= 1

        ego_vehicle_loc = self._vehicle.get_location()
        ego_vehicle_wp = self._map.get_waypoint(ego_vehicle_loc)

        print(f"Current speed: {self._speed:.2f} km/h")

        # In the following it is is managed the behavior of vehicles in critical situations (in order of priority):

        # 1. PEDESTRIAN HANDLER
        control = self.pedestrian_handler(ego_vehicle_wp, debug=debug)
        if control is not None:
            return control
        
        # 2. STOP SIGN HANDLER
        control = self.stop_sign_handler(debug=debug)
        if control is not None:
            return control

        # 3. TRAFFIC LIGHT HANDLER
        if self.traffic_light_handler():
            return self.emergency_stop() 
        
        # 4. JUNCTION HANDLER
        my_wp = self._map.get_waypoint(self._vehicle.get_location())
        curve_wp = my_wp.next(3.0)[0]
        control = self.junction_handler(ego_vehicle_loc, my_wp, curve_wp, debug=debug)
        if control is not None:
            return control

        # 5. VEHICLE HANDLER
        control = self.vehicle_handler(ego_vehicle_loc, ego_vehicle_wp, debug=debug)
        if control is not None:
            return control

        # 6. BIKE HANDLER
        control = self.bike_handler(ego_vehicle_wp, debug=debug)
        if control is not None:
            return control

        # 7. OBSTACLE HANDLER
        control = self.obstacle_handler(ego_vehicle_wp, debug=debug)
        if control is not None:
            return control

        # Normal driving (in the absence of the previous critical situations 1-7)
        if control is None:
            control = self.normal_driving(debug=debug)    
            self.last_steer = control.steer # saves the current steer (useful for overtaking bikes)

        return control

    #################### MAIN LOOP - END ####################
from local_planner import RoadOption
import carla
import math
from shapely.geometry import Polygon
from misc import get_speed, is_within_distance, get_trafficlight_trigger_location, compute_distance

class Perception:

    def __init__(self, behavior, world, vehicle, local_planner, map_inst=None):
        """
        Constructor method.
        """
        self._vehicle = vehicle
        self._world = self._vehicle.get_world()
        if map_inst:
            if isinstance(map_inst, carla.Map):
                self._map = map_inst
            else:
                print("Warning: Ignoring the given map as it is not a 'carla.Map'")
                self._map = self._world.get_map()
        else:
            self._map = self._world.get_map()
        self._last_traffic_light = None
    
        # Get the static elements of the scene
        self._bike_type_list = ['vehicle.gazelle.omafiets', 'vehicle.bh.crossbike', 'vehicle.diamondback.century']
        self._lights_map = {}  # Dictionary mapping a traffic light to a wp corrspoing to its trigger volume location

        # Parameters
        self._behavior = behavior
        self._world=world
        self._vehicle=vehicle
        self._local_planner=local_planner
        self._speed_limit=0
        self.current_speed = 0
        self._incoming_waypoint = None
        self._ignore_vehicles = False
        self._ignore_stop_signs = False
        self._base_vehicle_threshold = 5.0  # meters

    #################### UTILITY METHODS - START ####################     

    def _update_information(self, speed_limit, current_speed, look_ahead_steps=3):
        """
        This method updates the information regarding the ego vehicle based on the surrounding world.
        """
        self._speed_limit=speed_limit
        self.current_speed = current_speed
        self._incoming_waypoint, self._incoming_direction = self._local_planner.get_incoming_waypoint_and_direction(
            steps=look_ahead_steps)
        if self._incoming_direction is None:
            self._incoming_direction = RoadOption.LANEFOLLOW
        return
    
    def get_next_waypoint(self):
        ''' This method returns the next waypoint of the ego vehicle.'''
        return self._incoming_waypoint

    #################### UTILITY METHODS - END ####################


    #################### PEDESTRIAN PERCEPTION - START ####################

    def pedestrian_perception(self, waypoint,direction):
        """
        This method is responsible for the perception of pedestrians.
        """

        # Pedestrian perception
        pedestrian_list = self._world.get_actors().filter("*walker.pedestrian*")
        def dist(w): return w.get_location().distance(waypoint.transform.location)
        pedestrian_list = [w for w in pedestrian_list if dist(w) < 10]

        # Check the presence of pedestrians for the lane change
        if direction == RoadOption.CHANGELANELEFT:
            pedestrian_state, pedestrian, distance = self._pedestrian_obstacle_detected(pedestrian_list, max(
                self._behavior.min_proximity_threshold, self._speed_limit / 2), up_angle_th=90, lane_offset=-1)
        elif direction == RoadOption.CHANGELANERIGHT:
            pedestrian_state, pedestrian, distance = self._pedestrian_obstacle_detected(pedestrian_list, max(
                self._behavior.min_proximity_threshold, self._speed_limit / 2), up_angle_th=90, lane_offset=1)
        # Check the presence of pedestrians for lane following
        else:
            pedestrian_state, pedestrian, distance = self._pedestrian_obstacle_detected(pedestrian_list, max(
                self._behavior.min_proximity_threshold, self._speed_limit / 3), up_angle_th=60)

        return pedestrian_state, pedestrian, distance

    def _pedestrian_obstacle_detected(self, vehicle_list=None, max_distance=None, up_angle_th=90, low_angle_th=0, lane_offset=0):
        """
        Method to check if there is a pedestrian in front of the agent blocking its path.
        This method is similar to the _vehicle_obstacle_detected method, but it is specifically adapted for pedestrians.
        """
        
        if self._ignore_vehicles:
            return (False, None, -1)

        if not vehicle_list:
            vehicle_list = self._world.get_actors().filter(("*walker.pedestrian*"))

        if not max_distance:
            max_distance = self._base_vehicle_threshold

        ego_transform = self._vehicle.get_transform()
        ego_wpt = self._map.get_waypoint(self._vehicle.get_location())

        # Get the right offset
        if ego_wpt.lane_id < 0 and lane_offset != 0:
            lane_offset *= -1

        # Get the transform of the front of the ego
        ego_forward_vector = ego_transform.get_forward_vector()
        ego_extent = self._vehicle.bounding_box.extent.x
        ego_front_transform = ego_transform
        ego_front_transform.location += carla.Location(
            x=ego_extent * ego_forward_vector.x,
            y=ego_extent * ego_forward_vector.y,
        )

        for target_vehicle in vehicle_list:
            target_transform = target_vehicle.get_transform()
            target_wpt = self._map.get_waypoint(target_transform.location, lane_type=carla.LaneType.Any)

            # Simplified version for outside junctions
            if not ego_wpt.is_junction or not target_wpt.is_junction:
                if target_wpt.road_id != ego_wpt.road_id or target_wpt.lane_id != ego_wpt.lane_id  + lane_offset:
                    next_wpt = self._local_planner.get_incoming_waypoint_and_direction(steps=3)[0]
                    if not next_wpt:
                        continue
                    if target_wpt.road_id != next_wpt.road_id or target_wpt.transform.location.distance(next_wpt.transform.location) > 5.0:
                        continue

                target_forward_vector = target_transform.get_forward_vector()
                target_extent = target_vehicle.bounding_box.extent.x
                target_rear_transform = target_transform
                target_rear_transform.location -= carla.Location(
                    x=target_extent * target_forward_vector.x,
                    y=target_extent * target_forward_vector.y,
                )

                if is_within_distance(target_rear_transform, ego_front_transform, max_distance, [low_angle_th, up_angle_th]):
                    return (True, target_vehicle, compute_distance(target_transform.location, ego_transform.location))

            # Waypoints aren't reliable, check the proximity of the vehicle to the route
            else:
                route_bb = []
                ego_location = ego_transform.location
                extent_y = self._vehicle.bounding_box.extent.y
                r_vec = ego_transform.get_right_vector()
                p1 = ego_location + carla.Location(extent_y * r_vec.x, extent_y * r_vec.y)
                p2 = ego_location + carla.Location(-extent_y * r_vec.x, -extent_y * r_vec.y)
                route_bb.append([p1.x, p1.y, p1.z])
                route_bb.append([p2.x, p2.y, p2.z])

                for wp, _ in self._local_planner.get_plan():
                    if ego_location.distance(wp.transform.location) > max_distance:
                        break

                    r_vec = wp.transform.get_right_vector()
                    p1 = wp.transform.location + carla.Location(extent_y * r_vec.x, extent_y * r_vec.y)
                    p2 = wp.transform.location + carla.Location(-extent_y * r_vec.x, -extent_y * r_vec.y)
                    route_bb.append([p1.x, p1.y, p1.z])
                    route_bb.append([p2.x, p2.y, p2.z])

                if len(route_bb) < 3:
                    # 2 points don't create a polygon, nothing to check
                    return (False, None, -1)
                ego_polygon = Polygon(route_bb)

                # Compare the two polygons
                for target_vehicle in vehicle_list:
                    target_extent = target_vehicle.bounding_box.extent.x
                    if target_vehicle.id == self._vehicle.id:
                        continue
                    if ego_location.distance(target_vehicle.get_location()) > max_distance:
                        continue

                    target_bb = target_vehicle.bounding_box
                    target_vertices = target_bb.get_world_vertices(target_vehicle.get_transform())
                    target_list = [[v.x, v.y, v.z] for v in target_vertices]
                    target_polygon = Polygon(target_list)

                    if ego_polygon.intersects(target_polygon):
                        return (True, target_vehicle, compute_distance(target_vehicle.get_location(), ego_location))
                
                return (False, None, -1)
        
        return (False, None, -1)
    
    #################### PEDESTRIAN PERCEPTION - END ####################


    #################### STOP SIGN PERCEPTION - START ####################
    
    def stop_sign_perception(self):
        """
        This method is responsible for the perception of stop sign.
        """
        distance = min(max(((self.current_speed /3.6 )**2) / (2 * 5.21), 3), 5)
        actor_list = self._world.get_actors()
        stop_list = actor_list.filter("*stop*")
        affected, stop_sign_id = self.affected_by_stop_sign(stop_list, distance)
        return (affected,stop_sign_id)
    
    def affected_by_stop_sign(self, stop_list=None, distance_threshold=1.5):
        """
        Method to check if there is a stop sign affecting the vehicle.
        This method is similar to the _affected_by_traffic_light method, but it is specifically adapted for stop sign.
        """
        if self._ignore_stop_signs:
            return (False, -1)

        if not stop_list:
            stop_list = self._world.get_actors().filter("*stop*")

        ego_vehicle_location = self._vehicle.get_location()
        ego_vehicle_waypoint = self._map.get_waypoint(ego_vehicle_location)

        for stop_sign in stop_list:
            if stop_sign.id in self._lights_map:
                trigger_wp = self._lights_map[stop_sign.id]
            else:
                trigger_location = get_trafficlight_trigger_location(stop_sign)
                trigger_wp = self._map.get_waypoint(trigger_location)
                self._lights_map[stop_sign.id] = trigger_wp

            if trigger_wp.transform.location.distance(ego_vehicle_location) > distance_threshold:
                continue

            if trigger_wp.road_id != ego_vehicle_waypoint.road_id:
                continue

            ve_dir = ego_vehicle_waypoint.transform.get_forward_vector()
            wp_dir = trigger_wp.transform.get_forward_vector()
            dot_ve_wp = ve_dir.x * wp_dir.x + ve_dir.y * wp_dir.y + ve_dir.z * wp_dir.z

            if dot_ve_wp < 0:
                continue

            if is_within_distance(trigger_wp.transform, self._vehicle.get_transform(), distance_threshold, [0, 90]):
               return (True, stop_sign)

        return (False, -1)

    #################### STOP SIGN PERCEPTION - END ####################


    #################### VEHICLE PERCEPTION - START ####################

    def vehicle_perception(self, waypoint, distance=30):
        """
        This method is responsible for the perception of vehicle.
        """

        # Vehicle perception
        vehicle_list = self._world.get_actors().filter("*vehicle*")
        def dist(v): return v.get_location().distance(waypoint.transform.location)
        vehicle_list = [v for v in vehicle_list if dist(v) < distance and v.id != self._vehicle.id] 

        return vehicle_list

    #################### VEHICLE PERCEPTION - END ####################


    #################### BIKE PERCEPTION - START ####################

    def bike_perception(self, waypoint, direction):
        """
        This method is responsible for the perception of bikes.
        """

        # Bike perception        
        bike_list = self._world.get_actors().filter("*vehicle*")
        def dist(w): return w.get_location().distance(waypoint.transform.location)
        bike_list = [v for v in bike_list if dist(v) < 10 and v.id != self._vehicle.id and v.type_id in self._bike_type_list]
        
        # Check the presence of bikes for the lane change
        if direction == RoadOption.CHANGELANELEFT:
            bike_state, bike, b_distance = self._bike_obstacle_detected(bike_list, max(
                self._behavior.min_proximity_threshold, self._speed_limit / 2), up_angle_th=90, lane_offset=-1)
        elif direction == RoadOption.CHANGELANERIGHT:
            bike_state, bike, b_distance = self._bike_obstacle_detected(bike_list, max(
                self._behavior.min_proximity_threshold, self._speed_limit / 2), up_angle_th=90, lane_offset=1)
        # Check the presence of bikes for lane following
        else:
            bike_state, bike, b_distance = self._bike_obstacle_detected(bike_list, max(
                self._behavior.min_proximity_threshold, self._speed_limit / 3), up_angle_th=60)

        return bike_state, bike, b_distance

    def _bike_obstacle_detected(self, vehicle_list=None, max_distance=None, up_angle_th=90, low_angle_th=0, lane_offset=0):
        """
        Method to check if there is a bike in front of the agent blocking its path.
        This method is similar to the _vehicle_obstacle_detected method, but it is specifically adapted for bikes.
        """
        
        if self._ignore_vehicles:
            return (False, None, -1)

        if not vehicle_list:
            return (False, None, -1)
            
        if not max_distance:
            max_distance = self._base_vehicle_threshold

        ego_transform = self._vehicle.get_transform()
        ego_wpt = self._map.get_waypoint(self._vehicle.get_location())

        # Get the right offset
        if ego_wpt.lane_id < 0 and lane_offset != 0:
            lane_offset *= -1

        # Get the transform of the front of the ego
        ego_forward_vector = ego_transform.get_forward_vector()
        ego_extent = self._vehicle.bounding_box.extent.x
        ego_front_transform = ego_transform
        ego_front_transform.location += carla.Location(
            x=ego_extent * ego_forward_vector.x,
            y=ego_extent * ego_forward_vector.y,
        )

        for target_vehicle in vehicle_list:
            target_transform = target_vehicle.get_transform()
            target_wpt = self._map.get_waypoint(target_transform.location, lane_type=carla.LaneType.Any)

            # Simplified version for outside junctions
            if not ego_wpt.is_junction or not target_wpt.is_junction:
                if target_wpt.road_id != ego_wpt.road_id or target_wpt.lane_id != ego_wpt.lane_id  + lane_offset: #####
                    next_wpt = self._local_planner.get_incoming_waypoint_and_direction(steps=3)[0]
                    if not next_wpt:
                        continue
                    if target_wpt.road_id != next_wpt.road_id or target_wpt.transform.location.distance(next_wpt.transform.location) > 5.0:
                        continue

                target_forward_vector = target_transform.get_forward_vector()
                target_extent = target_vehicle.bounding_box.extent.x
                target_rear_transform = target_transform
                target_rear_transform.location -= carla.Location(
                    x=target_extent * target_forward_vector.x,
                    y=target_extent * target_forward_vector.y,
                )

                if is_within_distance(target_rear_transform, ego_front_transform, max_distance, [low_angle_th, up_angle_th]):
                    return (True, target_vehicle, compute_distance(target_transform.location, ego_transform.location))

            # Waypoints aren't reliable, check the proximity of the vehicle to the route
            else:
                route_bb = []
                ego_location = ego_transform.location
                extent_y = self._vehicle.bounding_box.extent.y
                r_vec = ego_transform.get_right_vector()
                p1 = ego_location + carla.Location(extent_y * r_vec.x, extent_y * r_vec.y)
                p2 = ego_location + carla.Location(-extent_y * r_vec.x, -extent_y * r_vec.y)
                route_bb.append([p1.x, p1.y, p1.z])
                route_bb.append([p2.x, p2.y, p2.z])

                for wp, _ in self._local_planner.get_plan():
                    if ego_location.distance(wp.transform.location) > max_distance:
                        break

                    r_vec = wp.transform.get_right_vector()
                    p1 = wp.transform.location + carla.Location(extent_y * r_vec.x, extent_y * r_vec.y)
                    p2 = wp.transform.location + carla.Location(-extent_y * r_vec.x, -extent_y * r_vec.y)
                    route_bb.append([p1.x, p1.y, p1.z])
                    route_bb.append([p2.x, p2.y, p2.z])

                if len(route_bb) < 3:
                    # 2 points don't create a polygon, nothing to check
                    return (False, None, -1)
                ego_polygon = Polygon(route_bb)

                # Compare the two polygons
                for target_vehicle in vehicle_list:
                    target_extent = target_vehicle.bounding_box.extent.x
                    if target_vehicle.id == self._vehicle.id:
                        continue
                    if ego_location.distance(target_vehicle.get_location()) > max_distance:
                        continue

                    target_bb = target_vehicle.bounding_box
                    target_vertices = target_bb.get_world_vertices(target_vehicle.get_transform())
                    target_list = [[v.x, v.y, v.z] for v in target_vertices]
                    target_polygon = Polygon(target_list)

                    if ego_polygon.intersects(target_polygon):
                        return (True, target_vehicle, compute_distance(target_vehicle.get_location(), ego_location))
                
                return (False, None, -1)
        
        return (False, None, -1)

    #################### BIKE PERCEPTION - END ####################


    #################### OBSTACLE PERCEPTION - START ####################

    def obstacle_perception(self, vehicle_list, max_distance=40,up_angle_th=90, low_angle_th=0, lane_offset=0):
        """
        method that returns the list of objects detected by the vehicle.
        This method is similar to the _vehicle_obstacle_detected method, but it is specifically adapted for obstacles.
        """
        
        if self._ignore_vehicles:
            return (False, None, -1)

        if not vehicle_list:
            vehicle_list = self._world.get_actors().filter("*static.prop*")

        if not max_distance:
            max_distance = self._base_vehicle_threshold

        ego_transform = self._vehicle.get_transform()
        ego_wpt = self._map.get_waypoint(self._vehicle.get_location())

        # Get the right offset
        if ego_wpt.lane_id < 0 and lane_offset != 0:
            lane_offset *= -1

        # Get the transform of the front of the ego
        ego_forward_vector = ego_transform.get_forward_vector()
        ego_extent = self._vehicle.bounding_box.extent.x
        ego_front_transform = ego_transform
        ego_front_transform.location += carla.Location(
            x=ego_extent * ego_forward_vector.x,
            y=ego_extent * ego_forward_vector.y,
        )

        obstacles=[]

        for target_vehicle in vehicle_list:
            target_transform = target_vehicle.get_transform()
            target_wpt = self._map.get_waypoint(target_transform.location, lane_type=carla.LaneType.Any)

            # Simplified version for outside junctions
            if not ego_wpt.is_junction or not target_wpt.is_junction:
                if target_wpt.road_id != ego_wpt.road_id or target_wpt.lane_id != ego_wpt.lane_id  + lane_offset:
                    next_wpt = self._local_planner.get_incoming_waypoint_and_direction(steps=3)[0]
                    if not next_wpt:
                        continue
                    if (target_wpt.road_id != next_wpt.road_id or target_wpt.lane_id != next_wpt.lane_id + lane_offset):
                         continue
                    
                    if get_speed(target_vehicle) !=0:
                        continue
                    
                target_forward_vector = target_transform.get_forward_vector()
                target_extent = target_vehicle.bounding_box.extent.x
                target_rear_transform = target_transform
                target_rear_transform.location -= carla.Location(
                    x=target_extent * target_forward_vector.x,
                    y=target_extent * target_forward_vector.y,
                )

                if is_within_distance(target_rear_transform, ego_front_transform, max_distance, [low_angle_th, up_angle_th]):
                    obstacles.append((True, target_vehicle, compute_distance(target_transform.location, ego_transform.location)))

        obstacles.sort(key=lambda x: x[2])         
        return obstacles

    def ahead_obstacle_detected(self, vehicle_list=None, max_distance=None, up_angle_th=90, low_angle_th=0, lane_offset=0):
        """
        Method to check if there is object or a vehicle in front of the agent blocking its path.
        This method is similar to the _vehicle_obstacle_detected method, but it is specifically adapted for obstacles.
        """
        
        if self._ignore_vehicles:
            return (False, None, -1)

        if not vehicle_list:
            return (False, None, -1)

        if not max_distance:
            max_distance = self._base_vehicle_threshold

        ego_transform = self._vehicle.get_transform()
        ego_wpt = self._map.get_waypoint(self._vehicle.get_location())

        # Get the right offset
        if ego_wpt.lane_id < 0 and lane_offset != 0:
            lane_offset *= -1

        # Get the transform of the front of the ego
        ego_forward_vector = ego_transform.get_forward_vector()
        ego_extent = self._vehicle.bounding_box.extent.x
        ego_front_transform = ego_transform
        ego_front_transform.location += carla.Location(
            x=ego_extent * ego_forward_vector.x,
            y=ego_extent * ego_forward_vector.y,
        )

        for target_vehicle in reversed(vehicle_list):
            target_transform = target_vehicle.get_transform()
            target_wpt = self._map.get_waypoint(target_transform.location, lane_type=carla.LaneType.Any)
            
            # Simplified version for outside junctions
            if not ego_wpt.is_junction or not target_wpt.is_junction:
                if target_wpt.road_id != ego_wpt.road_id or target_wpt.lane_id != ego_wpt.lane_id  + lane_offset:
                    next_wpt = self._local_planner.get_incoming_waypoint_and_direction(steps=3)[0]
                    if not next_wpt:
                        continue
                    if (target_wpt.road_id != next_wpt.road_id or target_wpt.lane_id != next_wpt.lane_id + lane_offset):
                         continue
                
                target_forward_vector = target_transform.get_forward_vector()
                target_extent = target_vehicle.bounding_box.extent.x
                target_rear_transform = target_transform
                target_rear_transform.location -= carla.Location(
                    x=target_extent * target_forward_vector.x,
                    y=target_extent * target_forward_vector.y,
                )

                if is_within_distance(target_rear_transform, ego_front_transform, max_distance, [low_angle_th, up_angle_th]):
                    return (True, target_vehicle, compute_distance(target_transform.location, ego_transform.location))

            # Waypoints aren't reliable, check the proximity of the vehicle to the route
            else:
                route_bb = []
                ego_location = ego_transform.location
                extent_y = self._vehicle.bounding_box.extent.y
                r_vec = ego_transform.get_right_vector()
                p1 = ego_location + carla.Location(extent_y * r_vec.x, extent_y * r_vec.y)
                p2 = ego_location + carla.Location(-extent_y * r_vec.x, -extent_y * r_vec.y)
                route_bb.append([p1.x, p1.y, p1.z])
                route_bb.append([p2.x, p2.y, p2.z])

                for wp, _ in self._local_planner.get_plan():
                    if ego_location.distance(wp.transform.location) > max_distance:
                        break

                    r_vec = wp.transform.get_right_vector()
                    p1 = wp.transform.location + carla.Location(extent_y * r_vec.x, extent_y * r_vec.y)
                    p2 = wp.transform.location + carla.Location(-extent_y * r_vec.x, -extent_y * r_vec.y)
                    route_bb.append([p1.x, p1.y, p1.z])
                    route_bb.append([p2.x, p2.y, p2.z])

                if len(route_bb) < 3:
                    # 2 points don't create a polygon, nothing to check
                    return (False, None, -1)
                ego_polygon = Polygon(route_bb)

                # Compare the two polygons
                for target_vehicle in vehicle_list:
                    target_extent = target_vehicle.bounding_box.extent.x
                    if target_vehicle.id == self._vehicle.id:
                        continue
                    if ego_location.distance(target_vehicle.get_location()) > max_distance:
                        continue

                    target_bb = target_vehicle.bounding_box
                    target_vertices = target_bb.get_world_vertices(target_vehicle.get_transform())
                    target_list = [[v.x, v.y, v.z] for v in target_vertices]
                    target_polygon = Polygon(target_list)

                    if ego_polygon.intersects(target_polygon):
                        return (True, target_vehicle, compute_distance(target_vehicle.get_location(), ego_location))
                
                return (False, None, -1)
        
        return (False, None, -1)
    
    def check_for_vehicle_on_overtaking_lane(self, waypoint, distance):
            """
            Method to check for the presence of a vehicle on the overtaking lane.
            """

            # Retrieve all vehicles in the simulation world
            danger_vehicle_list= self._world.get_actors().filter("*vehicle*")

            left_lane_wp = waypoint.get_left_lane()
            left_lane_id = left_lane_wp.lane_id

            danger_state = False # indicates if there is a vehicle in the overtaking lane
            def_danger_vehicle = False  # dangerous vehicle detected (if present)
            danger_distance = distance  # distance to the closest detected vehicle

            # Iterate over all vehicles in the environment
            for danger_vehicle in danger_vehicle_list:
                danger_transform = danger_vehicle.get_transform()
                danger_wpt = self._map.get_waypoint(danger_transform.location, lane_type=carla.LaneType.Any)
                
                # Check if the vehicle is ahead relative to the ego vehicle
                is_ahead=(danger_wpt.transform.location - waypoint.transform.location).dot(waypoint.transform.get_forward_vector()) > 0
                
                # Check if the vehicle is on the same road and in the overtaking lane
                if danger_wpt.road_id == waypoint.road_id and danger_wpt.lane_id == left_lane_id and is_ahead:
                    
                    # Verify whether the vehicle is within the specified distance and field of view
                    distance = danger_vehicle.get_location().distance(waypoint.transform.location)
                    if is_within_distance(danger_transform, waypoint.transform, danger_distance, [0, 90]):
                        danger_state = True
                        def_danger_vehicle = danger_vehicle
                        danger_distance = distance

            return (danger_state, def_danger_vehicle, danger_distance)

    def static_object_perception(self, ego_vehicle_wp):
        """
        This method is responsible for the perception of static object.
        """
        
        # Static object perception
        static_list = self._world.get_actors().filter("*static.prop*")
        def dist(v): return v.get_location().distance(ego_vehicle_wp.transform.location)
        static_list=[v for v in static_list if dist(v) < 45 and v.id != self._vehicle.id and not v.type_id in ["static.prop.dirtdebris01","static.prop.dirtdebris02","static.prop.dirtdebris03","static.prop.dirtdebris04"]]
        
        return static_list
    
    def cone_perception(self, ego_vehicle_wp):
        """
        This method is responsible for the perception of cones.
        """
        
        # Cone perception
        cone_list = self._world.get_actors().filter("*static.prop*")
        def dist(v): return v.get_location().distance(ego_vehicle_wp.transform.location)
        cone_list=[v for v in cone_list if dist(v) < 15 and v.type_id == "static.prop.constructioncone"]
        
        return cone_list

    #################### OBSTACLE PERCEPTION - END ####################

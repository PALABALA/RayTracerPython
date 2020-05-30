import numpy as np
import math
import random

nx = 200
ny = 100
sample = 50

origin = np.array([0, 0 , 0])
lower_left = np.array([-2, -1, -1])
horizontal = np.array([4, 0, 0])
vertical = np.array([0, 2, 0])


class Ray:
    def __init__(self, origin, direction):
        self.origin = origin
        self.direction = direction
        self.unit_direction = self.direction / np.sqrt(np.dot(self.direction, self.direction))

    def get_point_at_parameter(self, t):
        return self.origin + t * self.unit_direction


class Camera:
    def __init__(self, position, look_at, vup, vertical_fov, aspect_ratio, aperture, focus_dist):
        self.vertical_fov = vertical_fov
        self.aspect_ratio = aspect_ratio
        self.position = position
        self.lens_radius = aperture / 2

        theta = self.vertical_fov * math.pi / 180
        half_height = math.tan(theta / 2)
        half_width = self.aspect_ratio * half_height
        w = get_direction(self.position - look_at)
        self.u = get_direction(np.cross(vup, w))
        self.v = np.cross(w, self.u)

        self.lower_left_corner = self.position - half_width * focus_dist * self.u - half_height * focus_dist * self.v - focus_dist * w
        self.horizontal = 2 * half_width * focus_dist * self.u
        self.vertical = 2 * half_height * focus_dist * self.v

    def get_ray(self, s, t):
        rd = self.lens_radius * get_random_point_in_unit_disk()
        offset = self.u * rd[0] + self.v * rd[1]
        pixel_pos = self.lower_left_corner + s * self.horizontal + t * self.vertical
        origin = self.position + offset
        return Ray(origin, pixel_pos - origin)


def reflect(v, normal):
    return v - 2 * np.dot(v, normal) * normal


def refract(v, normal, ni_over_nt):
    v_normalized = get_direction(v)
    n_normalized = get_direction(normal)
    dt = np.dot(v_normalized, n_normalized)
    discriminant = 1 - ni_over_nt * ni_over_nt * (1 - dt * dt)
    if discriminant > 0:
        refracted = ni_over_nt * (v_normalized - n_normalized * dt) - n_normalized * math.sqrt(discriminant)
        return refracted
    return None


def schlick(cos, refraction_index):
    r0 = (1 - refraction_index) / (1 + refraction_index)
    r0 *= r0
    return r0 + (1 - r0) * ((1 - cos) ** 5)


def get_random_point_in_unit_sphere():
    while True:
        point = 2 * np.array([random.random(), random.random(), random.random()]) - np.array([1, 1, 1])
        if np.sqrt(np.dot(point, point)) < 1:
            return point


def get_random_point_in_unit_disk():
    while True:
        point = 2 * np.array([random.random(), random.random(), 0]) - np.array([1, 1, 0])
        if np.sqrt(np.dot(point, point)) >= 1:
            return point


def lerp(start, end, step):
    return (1 - step) * start + step * end


def get_direction(ray):
    return ray / np.sqrt(np.dot(ray, ray))


class HitRecord:
    def __init__(self, t=None, hitpoint=None, normal=None):
        self.t = t
        self.hitpoint = hitpoint
        self.normal = normal


class Sphere:
    def __init__(self, center, radius, material=None):
        self.center = center
        self.radius = radius
        self.material = material

    def hit(self, ray, t_min, t_max):
        ray_origin_to_sphere_center = ray.origin - self.center
        a = np.dot(ray.direction, ray.direction)
        b = np.dot(ray_origin_to_sphere_center, ray.direction)
        c = np.dot(ray_origin_to_sphere_center, ray_origin_to_sphere_center) - self.radius * self.radius
        discriminant = b * b - a * c
        if discriminant > 0:
            hit_record = HitRecord()
            t = (-b - math.sqrt(discriminant)) / a
            if t_min < t < t_max:
                hit_record.t = t
                hit_record.hitpoint = ray.get_point_at_parameter(hit_record.t)
                hit_record.normal = (hit_record.hitpoint - self.center) / self.radius
                return hit_record
            t = (-b + math.sqrt(discriminant)) / a
            if t_min < t < t_max:
                hit_record.t = t
                hit_record.hitpoint = ray.get_point_at_parameter(hit_record.t)
                hit_record.normal = (hit_record.hitpoint - self.center) / self.radius
                return hit_record
        return None


class Material:
    def __init__(self):
        pass

    def scatter(self, ray_in, hit_record):
        pass


class Lambert(Material):
    def __init__(self, albedo):
        self.albedo = albedo

    def scatter(self, ray_in, hit_record):
        bounce_target = hit_record.hitpoint + hit_record.normal + get_random_point_in_unit_sphere()
        scattered_ray = Ray(hit_record.hitpoint, bounce_target)
        attenuation = self.albedo
        return (scattered_ray, attenuation)


class Metal(Material):
    def __init__(self, albedo, fuzz):
        self.albedo = albedo
        self.fuzz = fuzz

    def scatter(self, ray_in, hit_record):
        reflected = reflect(ray_in.unit_direction, hit_record.normal)
        scattered_ray = Ray(hit_record.hitpoint, reflected + self.fuzz * get_random_point_in_unit_sphere())
        reflected_dir = get_direction(reflected)
        if np.dot(reflected_dir, hit_record.normal) <= 0:
            return (None, None)
        attenuation = self.albedo
        return (scattered_ray, attenuation)


class Dielectric(Material):
    def __init__(self, refraction_index):
        self.refraction_index = refraction_index # refraction index, 1.3 - 1.7 for glass

    def scatter(self, ray_in, hit_record):
        attenuation = np.array([1, 1, 1])
        ni_over_nt = None
        outward_normal = None
        cos = None

        if np.dot(ray_in.unit_direction, hit_record.normal) > 0:
            outward_normal = -hit_record.normal
            ni_over_nt = self.refraction_index
            cos = self.refraction_index * np.dot(ray_in.unit_direction, hit_record.normal) / np.sqrt(np.dot(ray_in.unit_direction, ray_in.unit_direction))
        else:
            outward_normal = hit_record.normal
            ni_over_nt = 1 / self.refraction_index
            cos = -np.dot(ray_in.unit_direction, hit_record.normal) / np.sqrt(np.dot(ray_in.unit_direction, ray_in.unit_direction))

        refracted = refract(ray_in.unit_direction, outward_normal, ni_over_nt)

        reflected = reflect(ray_in.unit_direction, hit_record.normal)

        if refracted is not None:
            reflect_prob = schlick(cos, self.refraction_index)
        else:
            reflect_prob = 1

        if random.random() < reflect_prob:
            scattered_ray = Ray(hit_record.hitpoint, reflected)
        else:
            scattered_ray = Ray(hit_record.hitpoint, refracted)
        return (scattered_ray, attenuation)


def hit_sphere_list(spheres, ray, min, max):
    closest_t = max
    closest_sphere = None
    closest_hit_record = None
    for sphere in spheres:
        hit_record = sphere.hit(ray, min, max)
        if hit_record and min < hit_record.t < closest_t:
            closest_t = hit_record.t
            closest_hit_record = hit_record
            closest_sphere = sphere
    return (closest_sphere, closest_hit_record)


def color(ray, spheres, bounce_count):
    if bounce_count > 50:
        return np.array([0, 0, 0])

    (sphere, hit_record) = hit_sphere_list(spheres, ray, 0.001, math.inf)

    if hit_record is not None:
        hitpoint = hit_record.hitpoint
        normal = hit_record.normal

        (scattered_ray, attenuation) = sphere.material.scatter(ray, hit_record)

        if scattered_ray is None:
            return np.array([0, 0, 0])

        return attenuation * color(scattered_ray, spheres, bounce_count + 1)

    t = 0.5 * (ray.unit_direction[1] + 1)
    return lerp(np.array([1, 1, 1]), np.array([0.5, 0.7, 1]), t)


with open('image.ppm', 'w') as out_file:
    out_file.write(f'P3\n{nx} {ny}\n255\n')

    sphere_r = math.cos(math.pi / 4)

    spheres = [
        Sphere(np.array([0, 0, -1]), 0.5, Lambert(np.array([0.1, 0.2, 0.5]))),
        Sphere(np.array([0, -100.5, -1]), 100, Lambert(np.array([0.8, 0.8, 0]))),
        Sphere(np.array([1, 0, -1]), 0.5, Metal(np.array([0.8, 0.6, 0.2]), 1)),
        Sphere(np.array([-1, 0, -1]), 0.5, Dielectric(1.5)), # refraction index, 1.3 - 1.7 for glass
        Sphere(np.array([-1, 0, -1]), -0.45, Dielectric(1.5))
    ]

    camera_pos = np.array([3, 3, 2])
    look_at = np.array([0, 0, -1])
    distance = camera_pos - look_at
    dist_to_focus = np.sqrt(np.dot(distance, distance))
    aperture = 2
    camera = Camera(camera_pos, look_at, np.array([0, 1, 0]), 20, nx / ny, aperture, dist_to_focus)

    for j in range(ny - 1, -1, -1):
        for i in range(nx):
            ray_color = np.array([0.0, 0.0, 0.0])
            for s in range(sample):
                u = (i + random.random()) / nx
                v = (j + random.random()) / ny
                ray = camera.get_ray(u, v)
                ray_color += color(ray, spheres, 0)
            ray_color /= sample
            ir = int(255.99 * math.sqrt(ray_color[0]))
            ig = int(255.99 * math.sqrt(ray_color[1]))
            ib = int(255.99 * math.sqrt(ray_color[2]))
            out_file.write(f'{ir} {ig} {ib}\n')
        print(j)

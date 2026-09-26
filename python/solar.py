from __future__ import annotations

from typing import Any
import time

import wgpu
from rendercanvas.glfw import RenderCanvas, loop

from camera2d import *
from transform import *
from node import *
from shader import *
from pipeline import *
from scene import *
from renderer import *
from engine import *
from disk import Disk
from square import Square
from texture import Texture
from textureset import TextureSet
from sampler import Sampler

canvas: RenderCanvas
device: wgpu.GPUDevice
context: Any
renderer: Renderer
camera: Camera2D
scene: Scene
last_t: float = 0.0


class RotateEngine(Engine):
    def __init__(self, trf: Transform, degrees_per_sec: float, axis: tuple[float, float, float] = (0, 0, -1)) -> None:
        self.trf = trf
        self.speed = degrees_per_sec
        self.axis = axis

    def update(self, dt: float) -> None:
        self.trf.rotate(self.speed * dt, *self.axis)


def initialize(device: wgpu.GPUDevice, target_format: str) -> None:
    global camera, scene

    camera = Camera2D(0, 10, 0, 10)

    shader = Shader(device, "../shaders/2d/textured.wgsl")
    shader.set_vertex_buffers([
        {"array_stride": 2 * 4, "step_mode": "vertex",
         "attributes": [{"format": "float32x2", "offset": 0, "var_name": "pos"}]},
        {"array_stride": 2 * 4, "step_mode": "vertex",
         "attributes": [{"format": "float32x2", "offset": 0, "var_name": "texcoord"}]},
    ])
    pipeline = Pipeline(shader, target_format, depth_stencil=None)

    sampler = Sampler(device, "decal_sampler")

    def make_texture_set(filename: str) -> TextureSet:
        tex = Texture(device, "decal_texture", f"../images/{filename}")
        ts = TextureSet([tex, sampler])
        shader.add_texture_set(ts)
        return ts

    space_ts = make_texture_set("space.avif")
    sun_ts = make_texture_set("sun.png")
    mercury_ts = make_texture_set("mercury.webp")
    earth_ts = make_texture_set("earth.jpg")
    moon_ts = make_texture_set("moon.png")

    trf_bg = Transform()
    trf_bg.translate(5, 5, -0.99)
    trf_bg.scale(20, 20, 1)
    background = Node(trf=trf_bg, apps=[space_ts], shps=[Square(device)])

    trf_sol = Transform()
    trf_sol.translate(5, 5, 0)
    trf_sol.scale(0.8, 0.8, 1)
    sol = Node(trf=trf_sol, apps=[sun_ts], shps=[Disk(device)])

    trf_mercury_orbit = Transform()
    trf_mercury_orbit.translate(5, 5, 0)

    trf_mercury_body = Transform()
    trf_mercury_body.translate(1.8, 0, 0)
    trf_mercury_body.scale(0.25, 0.25, 1)

    mercury_body = Node(trf=trf_mercury_body, apps=[mercury_ts], shps=[Disk(device)])
    mercury_orbit_node = Node(trf=trf_mercury_orbit, nodes=[mercury_body])

    trf_terra_orbit = Transform()
    trf_terra_orbit.translate(5, 5, 0)

    trf_terra_pos = Transform()
    trf_terra_pos.translate(3.0, 0, 0)

    trf_earth_spin = Transform()

    trf_terra_shape = Transform()
    trf_terra_shape.scale(0.45, 0.45, 1)

    terra_shape_node = Node(trf=trf_terra_shape, apps=[earth_ts], shps=[Disk(device)])
    earth_spin_node = Node(trf=trf_earth_spin, nodes=[terra_shape_node])

    trf_lua_orbit = Transform()
    trf_lua_pos = Transform()
    trf_lua_pos.translate(1.1, 0, 0)
    trf_lua_pos.scale(0.2, 0.2, 1)
    lua = Node(trf=trf_lua_pos, apps=[moon_ts], shps=[Disk(device)])

    lua_orbit_node = Node(trf=trf_lua_orbit, nodes=[lua])

    terra_pos_node = Node(trf=trf_terra_pos, nodes=[earth_spin_node, lua_orbit_node])
    terra_orbit_node = Node(trf=trf_terra_orbit, nodes=[terra_pos_node])

    root = Node(pipeline, nodes=[background, sol, mercury_orbit_node, terra_orbit_node])
    scene = Scene(root)

    scene.add_engine(RotateEngine(trf_mercury_orbit, 35))
    scene.add_engine(RotateEngine(trf_terra_orbit, 20))
    scene.add_engine(RotateEngine(trf_earth_spin, 90))
    scene.add_engine(RotateEngine(trf_lua_orbit, 90))


def update(dt: float) -> None:
    scene.update(dt)


def draw() -> None:
    global last_t
    t = time.perf_counter()
    update(t - last_t)
    last_t = t
    target_texture = context.get_current_texture()
    renderer.render(target_texture, scene, camera)


def on_key(event: Any) -> None:
    if event["key"] == "q":
        canvas.close()


def main() -> None:
    global canvas, device, context, renderer, last_t
    canvas = RenderCanvas(
        size=(600, 600),
        title="Mini Sistema Solar",
        update_mode="continuous",
        max_fps=60
    )
    adapter = wgpu.gpu.request_adapter_sync()
    device = adapter.request_device_sync()
    context = canvas.get_context("wgpu")
    target_format = context.get_preferred_format(device.adapter)
    context.configure(device=device, format=target_format)
    renderer = Renderer(device, clear_value=(0.0, 0.0, 0.05, 1.0))
    initialize(device, target_format)
    canvas.add_event_handler(on_key, "key_down")
    last_t = time.perf_counter()
    canvas.request_draw(draw)
    loop.run()


if __name__ == "__main__":
    main()

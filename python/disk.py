from __future__ import annotations
import math
from typing import TYPE_CHECKING
import numpy as np
import wgpu
from shape import Shape

if TYPE_CHECKING:
  from state import State

class Disk(Shape):
  def __init__(self, device: wgpu.GPUDevice, n: int = 64) -> None:
    coords = [[0.0, 0.0]]       
    texcoords = [[0.5, 0.5]]    

    for i in range(n):
      angle = 2 * math.pi * i / n
      x, y = math.cos(angle), math.sin(angle)
      coords.append([x, y])
      texcoords.append([(x + 1) / 2, (y + 1) / 2])  

    indices = []
    for i in range(1, n):
      indices += [0, i, i + 1]
    indices += [0, n, 1] 

    self.coord_vbo = device.create_buffer_with_data(
      data=np.array(coords, dtype='float32'), usage=wgpu.BufferUsage.VERTEX)
    self.tex_vbo = device.create_buffer_with_data(
      data=np.array(texcoords, dtype='float32'), usage=wgpu.BufferUsage.VERTEX)
    self.ibo = device.create_buffer_with_data(
      data=np.array(indices, dtype='uint32'), usage=wgpu.BufferUsage.INDEX)
    self.index_count = len(indices)

  def draw(self, st: State) -> None:
    first_instance = st.get_shader().commit_matrix(st)
    st.render_pass.set_vertex_buffer(0, self.coord_vbo)
    st.render_pass.set_vertex_buffer(1, self.tex_vbo)
    st.render_pass.set_index_buffer(self.ibo, wgpu.IndexFormat.uint32)
    st.render_pass.draw_indexed(self.index_count, 1, 0, 0, first_instance)
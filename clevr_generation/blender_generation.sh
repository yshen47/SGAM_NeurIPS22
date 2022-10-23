#!/bin/bash
for i in {1..20}
do
   blender -noaudio random_scene.blend --background --python blender_data_generation_grid.py
done

from nuscenes.nuscenes import NuScenes

nusc = NuScenes(version='v1.0-mini', dataroot='/media/oh/0E4A12890E4A1289/Nuimages-V1.0-mini', verbose=False)

### 1. `scene`

nusc.list_scenes()

my_scene = nusc.scene[0]




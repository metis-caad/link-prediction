# constants for room type
PARKING = 'PARKING'
BUILDINGSERVICES = 'BUILDINGSERVICES'
EXTERIOR = 'EXTERIOR'
STORAGE = 'STORAGE'
CHILDREN = 'CHILDREN'
TOILET = 'TOILET'
BATH = 'BATH'
CORRIDOR = 'CORRIDOR'
KITCHEN = 'KITCHEN'
WORKING = 'WORKING'
SLEEPING = 'SLEEPING'
LIVING = 'LIVING'
ROOM = 'ROOM'

# constants for edge type
WINDOW = 'WINDOW'
STAIRS = 'STAIRS'
SLAB = 'SLAB'
ENTRANCE = 'ENTRANCE'
WALL = 'WALL'
PASSAGE = 'PASSAGE'
DOOR = 'DOOR'
EDGE = 'EDGE'

# constants for zones
ZONE_SERVICE = 'ZONE_SERVICE'
ZONE_HABITATION = 'ZONE_HABITATION'
ZONE_SLEEPING = 'ZONE_SLEEPING'
ZONE_LIVING = 'ZONE_LIVING'
ZONE_DRY = 'ZONE_DRY'
ZONE_WET = 'ZONE_WET'

room_type_codes = {
    ROOM: 'r',
    LIVING: 'l',
    SLEEPING: 's',
    WORKING: 'w',
    KITCHEN: 'k',
    CORRIDOR: 'c',
    BATH: 'b',
    TOILET: 't',
    CHILDREN: 'h',
    STORAGE: 'g',
    EXTERIOR: 'e',
    BUILDINGSERVICES: 'v',
    PARKING: 'p'
}

room_types = {
    'l': '10',  # living
    's': '15',  # sleeping
    'w': '20',  # working
    'k': '25',  # kitchen
    'c': '30',  # corridor
    'b': '35',  # bath
    't': '40',  # toilet
    'h': '45',  # children
    'g': '50',  # storage
    'r': '55',  # room
    'e': '60',  # exterior
    'v': '65',  # buildingservices
    'p': '70'  # parking
}

room_code_numbers = {
    '10': LIVING,
    '15': SLEEPING,
    '20': WORKING,
    '25': KITCHEN,
    '30': CORRIDOR,
    '35': BATH,
    '40': TOILET,
    '45': CHILDREN,
    '50': STORAGE,
    '55': EXTERIOR,
    '60': ROOM,
    '65': BUILDINGSERVICES,
    '70': PARKING
}

edge_type_codes = {
    EDGE: 'e',
    DOOR: 'd',
    PASSAGE: 'p',
    WALL: 'w',
    ENTRANCE: 'r',
    SLAB: 'b',
    STAIRS: 's',
    WINDOW: 'n'
}

edge_types = {
    'd': '10',  # door
    'p': '15',  # passage
    'w': '20',  # wall
    'r': '25',  # entrance
    'b': '30',  # slab
    's': '35',  # stairs
    'n': '40',  # window
    'e': '45'  # edge
}

edge_code_numbers = {
    '10': DOOR,
    '15': PASSAGE,
    '20': WALL,
    '25': ENTRANCE,
    '30': SLAB,
    '35': STAIRS,
    '40': WINDOW,
    '45': EDGE
}

edge_type_weights = {
    EDGE: 0,
    DOOR: 1,
    PASSAGE: 1,
    ENTRANCE: 1,
    STAIRS: 1,
    WALL: 0.5,
    SLAB: 0.5,
    WINDOW: 0.5
}

all_zones = {
    ZONE_WET: [KITCHEN, TOILET, BATH],
    ZONE_DRY: [LIVING, SLEEPING, WORKING, CORRIDOR, CHILDREN],
    ZONE_LIVING: [KITCHEN, LIVING],
    ZONE_SLEEPING: [SLEEPING],
    ZONE_HABITATION: [KITCHEN, LIVING, SLEEPING, EXTERIOR, CHILDREN],
    ZONE_SERVICE: [CORRIDOR, TOILET, BATH, STORAGE, BUILDINGSERVICES, PARKING]
}

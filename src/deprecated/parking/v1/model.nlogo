extensions [nw csv table]   ;; "table" added in order to group the lot-list with utilities for all patches according to their lot-id

breed [nodes node]
breed [cars car]


globals
[
  grid-x-inc               ;; the amount of patches in between two roads in the x direction
  grid-y-inc               ;; the amount of patches in between two roads in the y direction
  acceleration             ;; the constant that controls how much a car speeds up or slows down by if
                           ;; it is to accelerate or decelerate
  intersec-max-x           ;; outer coordinates of intersections (required to set-up lots and garages)
  intersec-min-x
  intersec-max-y
  intersec-min-y

  speed-limit              ;; the maximum speed of the cars
  phase                    ;; keeps track of the phase (traffic lights)
  num-cars-stopped         ;; the number of cars that are stopped during a single pass thru the go procedure
  city-income              ;; money the city currently makes
  city-loss                ;; money the city loses because people do not buy tickets
  total-fines              ;; sum of all fines collected by the city
  lot-counter              ;; counter for id assigment
  num-spaces               ;; number of individual spaces
  n-cars                   ;; number of currently active cars
  mean-wait-time           ;; average wait time of cars
  yellow-lot-current-fee   ;; current fee of yellow
  green-lot-current-fee   ;; current fee of green
  teal-lot-current-fee    ;; current fee of teal
  blue-lot-current-fee     ;; current fee of blue
  potential-goals          ;; agents with all building patches

  yellow-lot-current-occup   ;; current occupation of yellow
  green-lot-current-occup   ;; current occupation of green
  teal-lot-current-occup    ;; current occupation of teal
  blue-lot-current-occup     ;; current occupation of blue
  garages-current-occup

  vanished-cars-low        ;; counter for cars that are not respawned
  vanished-cars-middle
  vanished-cars-high

  initial-count-low
  initial-count-middle
  initial-count-high

  global-occupancy         ;; overall occupancy of all lots
  cars-to-create           ;; number of cars that have to be created to replace those leaving the map
  low-to-create
  middle-to-create
  high-to-create
  mean-income              ;; mean income of turtles
  median-income            ;; median-income of turtles
  color-counter            ;; counter to ensure that every group of lots is visited only twice
  lot-colors               ;; colors to identify different lots

  selected-car   ;; the currently selected car //inserted

  ;; patch agentsets
  intersections ;; agentset containing the patches that are intersections
  roads         ;; agentset containing the patches that are roads
  yellow-lot    ;; agentset containing the patches that contain the spaces for the yellow lot
  teal-lot     ;; agentset containing the patches that contain the spaces for the teal lot
  green-lot    ;; agentset containing the patches that contain the spaces for the green lot
  blue-lot      ;; agentset containing the patches that contain the spaces for the blue lot
  lots          ;; agentset containing all patches that are roadside parking spaces
  lot-ids               ;; list of all lot ids of actual parking lots
  gateways      ;; agentset containing all patches that are gateways to garages
  garages       ;; agentset containing all patches that are parking spaces in garages
  park-spaces ;; Union of lots and garages
  finalpatches  ;; agentset containing all patches that are at the end of streets
  initial-spawnpatches ;; agentset containing all patches for initial spawning
  spawnpatches   ;;agentset containing all patches that are beginning of streets
  traffic-counter ;; counter to calibrate model to resemble subject as closely as possible
  income-entropy ;; normalized entropy of income-class distribution
  initial-low  ;; initial share of low income class
  normalized-share-low ;;
  mean-speed ;; average speed of cars not parking
  share-cruising ;; share of cars crusing

  ;; MO
  original-yellow-lot
  original-teal-lot
  original-green-lot
  original-blue-lot
  original-lots
  original-garages
  original-park-spaces
  allowed-parking-spaces

  one-strategy-change-counter    ;; axhausen: total counter for one strategy changes;; modeling tipping point at around 400 traffic count density leading to rapid strategy changing
  two-strategy-change-counter    ;; axhausen: total counter for two strategy changes;; after that traffic count, the agent stuck in traffic change strategies
  road-space-dict                  ;; dict used to look-up patches between current street and parking space of choice
  synthetic-population            ;;list of turtles based on synthetic data
                                  ;;debugging

  ;; MO
  output-traffic-file-path
  road-sections
]

nodes-own
[

]
cars-own
[
  wait-time ;; time passed since a turtle has moved

  speed         ;; the current speed of the car
  park-time     ;; the time the driver wants to stay in the parking-spot
  park          ;; the driver's probability to be searching for a parking space
  wants-to-park ;; boolean to indicate whether the driver searches for parking
  paid?         ;; true if the car paid for its spot
  looks-for-parking? ;; whether the agent is currently looking for parking in the target street
  parked?       ;; true if the car is parked
  time-parked     ;; time spent parking
  final-access
  computed-access
  final-egress
  final-type

  income          ;; income of agent
  gender          ;; gender of the agent
  age             ;; age of the agent
  parking-strategy ;; parking stratey of the agent
  school-degree   ;; school degree of the agent
  degree          ;; subsequent degree of the agent
  purpose         ;; purpose of visit (job/education, doctor, acquaintance, shopping)
  wtp             ;; Willigness to Pay for parking
  wtp-increased   ;; counter to keep track of how often the wtp was increased when searching
  income-interval-survey ;; income interval as reported in our survey
  parking-offender? ;; boolean for parking offenders
  lots-checked     ;; patch-set of lots checked by the driver
  direction-turtle ;; turtle dircetion
  nav-goal         ;; objective of turtle on the map
  nav-prklist      ;; list of parking spots sorted by distance to nav-goal
  nav-hastarget?   ;; boolean to check if agent has objective
  nav-pathtofollow ;; list of nodes to follow
  income-group     ;; ordinial classification of income
  search-time    ;; time to find a parking space
  reinitialize? ;; for agents who have left the map
  die?
  alternative? ;; boolean for compute-alterantive-route
  fee-income-share ;; fee as a portion of income
  distance-parking-target ;; distance from parking to target
  max-dist-parking-target   ;; maximal distance between parking lot and target
  max-dist-location-parking ;; maximal distance between agent and target
  price-paid       ;; price paid for pricing
  expected-fine    ;; expected fine for parking offender
  outcome          ;; outcome of agents (utility achieved after parking)

  utility-value             ;; utility calculated for agent
  temp-utility-value        ;; helper for utlity calculation

  informed-flag             ;; polak: flag indicating whether agent is informed or not, random assignment
  agent-strategy-flag       ;; polak: flag indicating which strategy agent selects based on whether its informed or not, Polak et al.
  hard-weight-list          ;; polak: binary weight list based on the selected parking strategy for the utility function
  fuzzy-weight-list         ;; polak: fuzzy weight list based on the selected parking strategy for the utility function
  logit-weights             ;; logit weights based on surve
  switch-strategy-flag      ;; axhausen: strategy changing flag, swap based on 'wait-time' value, when it becomes 1.5x & 2x times the 'mean-wait-time' value

  util-increase             ;; counter to keep track of how often a parking spot was not used

  fav-lot-id                ;; current parking target according to maximum utility
  fav-lot-ids
]

patches-own
[
  road?           ;; true if the patch is a road
  road-section    ;; id for road sections between intersections
  horizontal?     ;; true for road patches that are horizontal; false for vertical roads
                  ;;-1 for non-road patches
  alternate-direction? ;; true for every other parallel road.
                       ;;-1 for non-road patches
  direction       ;; one of "up", "down", "left" or "right"
                  ;;-1 for non-road patches
  intersection?   ;; true if the patch is at the intersection of two roads
  park-intersection? ;; true if the intersection has parking spaces
  dirx
  diry
  green-light-up? ;; true if the green light is above the intersection.  otherwise, false.
                  ;; false for a non-intersection patches.
  my-row          ;; the row of the intersection counting from the upper left corner of the
                  ;; world.  -1 for non-intersection and non-horizontal road patches.
  my-column       ;; the column of the intersection counting from the upper left corner of the
                  ;; world.  -1 for non-intersection and non-vertical road patches.
  my-phase        ;; the phase for the intersection.  -1 for non-intersection patches.
  car?            ;; whether there is a car on this patch
  fee             ;; price of parking here
  group-fees     ;;income-specific prices of parking
  lot-id          ;; id of lot
  center-distance ;; distance to center of map
  garage?         ;; true for private garages
  gateway?        ;; true for gateways of garages

  service?                 ;; determines whether the parking spot is secure (1) or not (0)

  ;; MO
  allowed? 
]


;;;;;;;;;;;;;;;;;;;;;;
;; Setup Procedures ;;
;;;;;;;;;;;;;;;;;;;;;;

;; Initialize the display by giving the global and patch variables initial values.
;; Create num-cars of turtles if there are enough road patches for one turtle to
;; be created per road patch. Set up the plots.
to setup
  random-seed 0
  
  clear-all
  reset-ticks
  reset-timer
  let start-time timer
  setup-globals
  set speed-limit 0.9  ;;speed required for somewhat accurate representation

  ;; First we ask the patches to draw themselves and set up a few variables
  setup-patches
  ;; set demand appropriate for 8:00 A.M.
  set parking-cars-percentage ((-5.58662028e-04 * 8 ^ 3 + 2.76514862e-02 * 8 ^ 2 + -4.09343614e-01 *  8 +  2.31844786e+00)  + demand-curve-intercept) * 100

  set-default-shape cars "car top"

  if (num-cars > count roads)
  [
    user-message (word "There are too many cars for the amount of "
      "road.  Either increase the amount of roads "
      "by increasing the GRID-SIZE-X or "
      "GRID-SIZE-Y sliders, or decrease the "
      "number of cars by lowering the NUMBER slider.\n"
      "The setup has stopped.")
    stop
  ]

  ;; Now create the turtles and have each created turtle call the functions setup-cars and set-car-color
  ;; load data for turtles from synthetic population
  if use-synthetic-population [
    ;; index,gender,age,household_income,degree,school_degree,parking_strategy,income,income_group
    set synthetic-population remove-item 0 csv:from-file synthetic-population-file-path ;; remove col names
  ]
  create-cars num-cars
  [
    setup-cars
    set-car-color
    record-data
    ifelse wants-to-park [
      setup-parked
    ]
    [
      set nav-prklist [] ;; let the rest leave the map
    ]
  ]
  if ((count cars with [wants-to-park]) / num-spaces) < target-start-occupancy
  [
    user-message (word "There are not enough cars to meet the specified "
      "start target occupancy rate.  Either increase the number of roads "
      ", or decrease the "
      "number of parking spaces by by lowering the lot-distribution-percentage slider.")
  ]

  ;; give the turtles an initial speed
  ask cars [ set-car-speed ]

  if demo-mode [ ;; for demonstration purposes
    let example_car one-of cars with [wants-to-park and not parked?]
    ask example_car [
      set color cyan
      set nav-prklist find-favorite-space    ;; polak: passing 'fuzzy-weight-list' values into the 'navigate' function
      set park parking-cars-percentage / 2
    ]
    watch example_car
    inspect example_car
    ask [nav-goal] of example_car [set pcolor cyan]
  ]

  ;; for Reinforcement Learning reward function
  set initial-low count cars with [income-group = 0] / count cars
  ;; for normalization of vanished cars plot
  set initial-count-low count cars with [income-group = 0]
  set initial-count-middle count cars with [income-group = 1]
  set initial-count-high count cars with [income-group = 2]

  record-globals
  ;; for documentation of all agents at every timestep
  if document-turtles [
    file-open output-turtle-file-path
    file-print csv:to-row (list "id" "income" "income-group" "income-interval-survey" "age" "gender" "school-degree" "degree" "wtp" "parking-strategy" "purpose" "parking-offender?" "checked-blocks" "egress" "access" "space-type" "price-paid" "search-time" "wants-to-park" "die?" "reinitialize?" "outcome")
  ]

  ;; MO
  set output-traffic-file-path "traffic.csv"
  set road-sections remove-duplicates [road-section] of roads
  file-open output-traffic-file-path
  file-print csv:to-row (list "ticks" "road-section" "number-of-cars" "mean-speed")


  let end-time timer
  if debugging [show end-time - start-time]
end

;; identifies patches on which cars die
to setup-finalroads
  set finalpatches roads with [(pxcor = max-pxcor and direction = "right") or (pxcor = min-pxcor and direction = "left") or (pycor  = max-pycor and direction = "up") or (pycor = min-pycor and direction = "down") ]

end

;; identifies roads on which cars are allowed to be spawned
to setup-spawnroads
  set spawnpatches roads with [(pxcor = max-pxcor and direction = "left") or (pxcor = min-pxcor and direction = "right") or (pycor  = max-pycor and direction = "down") or (pycor = min-pycor and direction = "up") ]
end

;; spawn intial cars so that they can navigate over the map
to setup-initial-spawnroads
  set initial-spawnpatches roads with [not intersection?]
end



;; Initialize the global variables to appropriate values
to setup-globals
  set phase 0
  set num-cars-stopped 0
  set grid-x-inc 15
  set grid-y-inc floor(grid-x-inc * 1.43)

  set n-cars num-cars

  set vanished-cars-low 0
  set vanished-cars-middle 0
  set vanished-cars-high 0


  set traffic-counter 0

  set mean-income 0
  set median-income 0
  set n-cars 0
  set mean-wait-time 0
  set mean-speed 0

  set yellow-lot-current-fee 0
  set green-lot-current-fee 0
  set teal-lot-current-fee 0
  set blue-lot-current-fee 0

  set global-occupancy 0

  set yellow-lot-current-occup 0
  set green-lot-current-occup 0
  set teal-lot-current-occup 0
  set blue-lot-current-occup 0
  if num-garages > 0 [set garages-current-occup 0]
  set income-entropy 0
  set initial-low 0
  set normalized-share-low 0
  set share-cruising 0

  ;; don't make acceleration 0.1 since we could get a rounding error and end up on a patch boundary
  set acceleration 0.099

  ;;debugging

  ;reset-timer
end


;; Make the patches have appropriate colors, set up the roads, parking space and intersections agentsets,
;; and initialize the traffic lights to one setting
to setup-patches
  ;; initialize the patch-owned variables and color the patches to a base-color
  ask patches
  [
    set road? false
    set horizontal? -1
    set alternate-direction? -1
    set direction -1
    set intersection? false
    set green-light-up? true
    set my-row -1
    set my-column -1
    set my-phase -1
    set pcolor [221 218 213]
    set center-distance [distancexy 0 0] of self
  ]

  ;; initialize the global variables that hold patch agentsets

  set roads patches with
    [(floor((pxcor + max-pxcor - floor(grid-x-inc - 1)) mod grid-x-inc) = 8) or
      (floor((pycor + max-pycor) mod grid-y-inc) = 8)]
  setup-roads
  set intersections roads with
    [(floor((pxcor + max-pxcor - floor(grid-x-inc - 1)) mod grid-x-inc) = 8) and
      (floor((pycor + max-pycor) mod grid-y-inc) = 8)]

  set intersec-max-x max [pxcor] of intersections
  set intersec-min-x min [pxcor] of intersections
  set intersec-max-y max [pycor] of intersections
  set intersec-min-y min [pycor] of intersections

  setup-intersections
  setup-lots
  if num-garages > 0 [setup-garages]
  setup-finalroads
  setup-spawnroads
  setup-initial-spawnroads
  setup-nodes
  setup-road-sections

  ;; all non-road patches can become goals
  set potential-goals patches with [pcolor = [221 218 213]]
end

;; creates roads in the model
to setup-roads
  ask roads [
    set road? true
    set pcolor white
    ;; check if patches left and right (x +-1 road?) are patches if yes, then it is a horizontal road
    ifelse (floor((pxcor + max-pxcor - floor(grid-x-inc - 1)) mod grid-x-inc) = 8)
    [set horizontal? false];; vertical road
    [set horizontal? true];; horizontal road

    ifelse horizontal?
    [ ;; horizontal roads get the row set
      set my-row floor((pycor + max-pycor) / grid-y-inc)
      ifelse my-row mod 2 = 1 ;; every other horizontal road has an alternate direction: normal + horizontal = right
      [ set alternate-direction? false
        set dirx "right"]
      [ set alternate-direction? true
        set dirx "left"]
      set direction dirx
    ]
    [ ;; vertial roads get the row set
      set my-column floor((pxcor + max-pxcor) / grid-x-inc)
      ifelse my-column mod 2 = 1 ;; every other vertial road has an alternate direction: normal + vertical = down
      [ set alternate-direction? true
        set diry "up"]
      [ set alternate-direction? false
        set diry "down"]
      set direction diry
    ]


    sprout-nodes 1 [ ;; node network for navigation
      set size 0.2
      set shape "circle"
      set color white
    ]


  ]

end

;; allows for identifying different streets and creates dictionary to save distances between streets and
;; traffic on said streets (is computed every half hour simulated)allows for identifying different streets and creates dictionary to save distances between streets and traffic on said streets (is computed every half hour simulated)
to setup-road-sections
  let id 1
  let roads-to-id roads with  [road-section = 0]
  ask intersections [
    let x [pxcor] of self
    let y [pycor] of self
    ask roads-to-id with [pxcor = x and pycor > y and pycor <= y + 20] [
      set road-section id
    ]
    set id id + 1
    ask roads-to-id with [pxcor = x and pycor < y and pycor >= y - 20] [
      set road-section id
    ]
    set id id + 1
    ask roads-to-id with [pycor = y and pxcor < x and pxcor >= x - 14] [
      set road-section id
    ]
    set id id + 1
    ask roads-to-id with [pycor = y and pxcor > x and pxcor <= x + 14] [
      set road-section id
    ]
    set id id + 1
    set roads-to-id roads-to-id with  [road-section = 0]
  ]

  set road-space-dict table:make
  foreach (remove-duplicates [road-section] of roads)
  [ section ->
    table:put road-space-dict section table:make
    let road-patch one-of roads with [road-section = section]
    let local-node one-of nodes-on road-patch
    let furthest-space max-one-of park-spaces [get-path-length (local-node) (get-closest-node lot-id)]
    table:put (table:get road-space-dict section) "max-dist-parking-target" get-path-length (local-node) (get-closest-node [lot-id] of furthest-space)
    ask local-node [
      foreach (remove-duplicates [lot-id] of park-spaces)  [space ->
        ;let lot one-of lots with [lot-id = id]
        table:put (table:get road-space-dict section) space table:make
        let node-proxy get-closest-node space
        let nodes-on-path (turtle-set nw:turtles-on-path-to node-proxy)
        let patches-on-path (patch-set [patch-here] of nodes-on-path)
        table:put (table:get (table:get road-space-dict section) space) "patches" (patches-on-path)
        table:put (table:get (table:get road-space-dict section) space) "path-length"  (count patches-on-path)
        table:put (table:get (table:get road-space-dict section) space) "traffic" count cars-on patches-on-path / (count  patches-on-path)
      ]
    ]
  ]
end

;; creates nodes for navigation
to setup-nodes
  ask nodes [
    (ifelse
      dirx = "left" [create-links-to nodes-on patch-at -1 0[set hidden? hide-nodes]]
      dirx = "right" [create-links-to nodes-on patch-at 1 0[set hidden? hide-nodes] ])

    (ifelse
      diry = "up" [create-links-to nodes-on patch-at 0 1[set hidden? hide-nodes]]
      diry = "down" [create-links-to nodes-on patch-at 0 -1[set hidden? hide-nodes] ])
  ]
end


;; Give the intersections appropriate values for the intersection?, my-row, and my-column
;; patch variables.  Make all the traffic lights start off so that the lights are red
;; horizontally and green vertically.
to setup-intersections
  ask intersections
  [
    set intersection? true
    set green-light-up? true
    set my-phase 0
    set my-row floor((pycor + max-pycor) / grid-y-inc)
    ifelse my-row mod 2 = 1 ;; every other horizontal road has an alternate direction: normal + horizontal = right
      [ set dirx "right"]
    [ set dirx "left"]
    set my-column floor((pxcor + max-pxcor) / grid-x-inc)
    ifelse my-column mod 2 = 1 ;; every other vertial road has an alternate direction: normal + vertical = down
      [ set diry "up"]
    [ set diry "down"]
    set-signal-colors
  ]
end

;; creates curbside parking spaces (somewhat randomly) and designates different Controlled Parking Zones (CPZs)
to setup-lots;;intialize dynamic lots
  set lot-counter 1
  ask intersections [set park-intersection? false]

  let potential-intersections intersections
  ask n-of (count potential-intersections * lot-distribution-percentage) potential-intersections [set park-intersection? true] ;; create as many parking lots as specified by lot-distribution-percentage  variable
                                                                                                                               ;; check if there is enough space for garages
  let garage-intersections intersections with [not park-intersection? and pxcor != intersec-max-x and pycor != intersec-min-y and pycor != intersec-min-y + grid-y-inc] ;; second intersec from down-left cannot be navigated
  if num-garages > 0 and num-garages > count garage-intersections[
    user-message (word "There are not enough free intersections to create the garages. "
      "Decrease the lot-occupancy to create the neccessary space. "
      "For this simulation, the number of on-street lots will be decreased.")
    ask n-of (num-garages) intersections with [park-intersection? = true and pxcor != intersec-max-x and pycor != intersec-min-y and pycor != intersec-min-y + grid-y-inc] [
      set park-intersection? false
    ]
  ]
  ask intersections with [park-intersection? = true][
    let x [pxcor] of self
    let y [pycor] of self
    if x != intersec-max-x and x != intersec-min-x and y != intersec-max-y and y != intersec-min-y [ ;;lots at the beginning and end of grid do not work with navigation
      spawn-lots x y "all"
    ]
    if x = intersec-min-x and y != intersec-min-y [ ;; create all possible lots on the right for the intersections that are on the left border
      spawn-lots x y "all"
    ]
    if x = intersec-max-x and y != intersec-min-y and y != intersec-max-y [ ;; create only down lots for the intersections that are on the right border
      spawn-lots x y "down"
    ]
    if y = intersec-max-y and x != intersec-max-x and x != intersec-min-x [ ;; create all lots below for the intersections that are on the upper border
      spawn-lots x y "all"
    ]
    if y = intersec-min-y and x < intersec-max-x [ ;; create only lower lots for intersections on lower border
      spawn-lots x y "right"
    ]
  ]

  ;; create patch-sets for different parking zones by distance to center of map
  set yellow-lot no-patches
  set green-lot no-patches
  set teal-lot no-patches
  set blue-lot no-patches

  let lot-distances sort remove-duplicates [center-distance] of patches with [lot-id != 0]
  let lot-count length lot-distances
  let i 0
  foreach lot-distances [lot-distance ->
    if i <= lot-count * 0.1[
      set yellow-lot (patch-set yellow-lot patches with [lot-id != 0 and center-distance = lot-distance])
    ]
    if i > lot-count * 0.1 and i <= lot-count * 0.35[
      set green-lot (patch-set green-lot patches with [lot-id != 0 and center-distance = lot-distance])
    ]
    if i > lot-count * 0.35 and i <= lot-count * 0.6[
      set teal-lot (patch-set teal-lot patches with [lot-id != 0 and center-distance = lot-distance])
    ]
    if i > lot-count * 0.6[
      set blue-lot (patch-set blue-lot patches with [lot-id != 0 and center-distance = lot-distance])
    ]
    set i i + 1
  ]

  ;; create patch-set for all parking spaces on the curbside
  set lots (patch-set yellow-lot teal-lot green-lot blue-lot)
  set num-spaces count lots

  set lot-ids [lot-ids] of lots;; list of all possible lot-ids being actual parking lots

  ;; MO
  set original-yellow-lot patch-set yellow-lot
  set original-green-lot patch-set green-lot
  set original-teal-lot patch-set teal-lot
  set original-blue-lot patch-set blue-lot
  set original-lots (patch-set original-yellow-lot original-teal-lot original-green-lot original-blue-lot)
  ask lots [set allowed? true]

  ;; color parking zones
  let yellow-c [255 255 102]
  ask yellow-lot [
    set pcolor yellow-c
    set fee yellow-lot-fee
    if group-pricing [set group-fees (list yellow-lot-fee yellow-lot-fee yellow-lot-fee)]
  ]
  let green-c [123 174 116]
  ask green-lot [
    set pcolor green-c
    set fee green-lot-fee
    if group-pricing [set group-fees (list green-lot-fee green-lot-fee green-lot-fee)]
  ]
  let teal-c [57 107 148]
  ask teal-lot [
    set pcolor teal-c
    set fee teal-lot-fee
    if group-pricing [set group-fees (list teal-lot-fee teal-lot-fee teal-lot-fee)]
  ]
  let blue-c 	[26 51 179]
  ask blue-lot [
    set pcolor blue-c
    set fee blue-lot-fee
    if group-pricing [set group-fees (list blue-lot-fee blue-lot-fee blue-lot-fee)]
  ]

  set lot-colors (list yellow-c green-c teal-c blue-c) ;; will be used to identify the different zones
end

;; creates lots, specification controls whether only to the right or down of intersection (or both)
to spawn-lots [x y specification] ;;
  let right-lots false
  let down-lots false
  ifelse specification = "all" [
    set right-lots true
    set down-lots true
  ]
  [
    ifelse specification = "right"[
      set right-lots true
    ]
    [
      set down-lots true
    ]
  ]
  if down-lots [
    ifelse random 100 >= 25 [ ;; random variable so that in 75% of cases, parking spots on both sides of road are created
      let potential-lots patches with [((pxcor = x + 1 ) or (pxcor = x - 1)) and ((pycor >= y - ( grid-y-inc * .75)) and (pycor <= y - (grid-y-inc * .25)))]
      let average-distance mean [center-distance] of potential-lots
      ask potential-lots [
        set center-distance average-distance ;;assign lot the average distance of its members
        set lot-id lot-counter
      ]
      set lot-counter lot-counter + 1
    ]
    [
      let random-x ifelse-value (random 100 <= 50) [1] [-1]
      let potential-lots patches with [((pxcor = x + random-x)) and ((pycor >= y - ( grid-y-inc * .75)) and (pycor <= y - (grid-y-inc * .25)))]
      let average-distance mean [center-distance] of potential-lots
      ask potential-lots [
        set center-distance average-distance
        set lot-id lot-counter
      ]
      set lot-counter lot-counter + 1
    ]
  ]
  if right-lots [
    ifelse random 100 >= 25 [
      let potential-lots patches with [((pycor = y + 1 ) or (pycor = y - 1)) and (((pxcor <= x + ( grid-x-inc * .75)) and (pxcor >= x + (grid-x-inc * .25))))]
      let average-distance mean [center-distance] of potential-lots
      ask potential-lots[
        set center-distance average-distance
        set lot-id lot-counter
      ]
      set lot-counter lot-counter + 1
    ]
    [
      let random-y ifelse-value (random 100 <= 50) [1] [-1]
      let potential-lots patches with [(pycor = y + random-y) and (((pxcor <= x + ( grid-x-inc * .75)) and (pxcor >= x + (grid-x-inc * .25))))]
      let average-distance mean [center-distance] of potential-lots
      ask potential-lots [
        set center-distance average-distance
        set lot-id lot-counter
      ]
      set lot-counter lot-counter + 1
    ]
  ]
end

;; create garages
to setup-garages
  ask patches [
    set garage? false
    set gateway? false
  ]
  let garage-intersections n-of (num-garages) intersections with [not park-intersection? and pxcor != intersec-max-x and pycor != intersec-min-y] ;; second intersec from down-left cannot be navigated
  ask garage-intersections[
    let x [pxcor] of self
    let y [pycor] of self
    let dir-intersec [direction] of self
    let potential-garages patches with [(((pxcor <= x + ( grid-x-inc * .7)) and (pxcor >= x + (grid-x-inc * .25)))) and ((pycor >= y - ( grid-y-inc * .7)) and (pycor <= y - (grid-y-inc * .25)))]
    let id (max [lot-id] of patches) + 1
    ask potential-garages [
      set pcolor [0 0 0]
      set direction dir-intersec
      set lot-id id
      set fee 2
      set garage? true
      ask patches with [((pxcor <= x + ( grid-x-inc * .25)) and (pxcor > x )) and (pycor = floor(y - ( grid-y-inc * .5)))] [
        set pcolor [0 0 0]
        if [pxcor] of self = x + 1[
          set gateway? true
          set lot-id id
        ]
      ]
    ]
  ]
  set garages patches with [garage?]
  set gateways patches with [gateway?]
  set park-spaces (patch-set lots garages)

  ;; MO
  set original-garages patch-set garages
  set original-park-spaces patch-set park-spaces
  ask garages [set allowed? true]
end

;; Initialize the turtle variables to appropriate values and place the turtles on an empty road patch.
to setup-cars  ;; turtle procedure
               ;show ""
  set speed 0
  set wait-time 0

  ;; check whether agent is created at beginning of model (reinitialize? = 0) or recreated during run of simulation (reinitialize? = true)
  ifelse reinitialize? = 0 [
    put-on-empty-road
    ifelse use-synthetic-population [
      draw-synthetic-turtle
      ;; always shuffle after drawing a turtle
      set synthetic-population shuffle synthetic-population
    ]
    [
      set income draw-income
    ]
    set income-group find-income-group
    set park random 100
  ]
  [
    ifelse count spawnpatches with [not any? cars-on self] > 0
    [
      move-to one-of spawnpatches with [not any? cars-on self]
    ]
    [
      put-on-empty-road
    ]
    ;; income of recreated cars is based on the distro of the model
    (ifelse
      low-to-create > 0 [
        set low-to-create low-to-create - 1
        set income-group 99
        ;; not very efficient, draw until desired income is drawn
        while [income-group != 0] [
          ifelse use-synthetic-population [
            draw-synthetic-turtle
            set synthetic-population shuffle synthetic-population
          ]
          [
            set income draw-income
          ]
          set income-group find-income-group
        ]
      ]
      middle-to-create > 0 [
        set middle-to-create middle-to-create - 1
        set income-group 99
        while [income-group != 1] [
          ifelse use-synthetic-population [
            draw-synthetic-turtle
            set synthetic-population shuffle synthetic-population
          ]
          [
            set income draw-income
          ]
          set income-group find-income-group
        ]
      ]
      high-to-create > 0 [
        set high-to-create high-to-create - 1
        set income-group 99
        while [income-group != 2] [
          ifelse use-synthetic-population [
            draw-synthetic-turtle
            set synthetic-population shuffle synthetic-population
          ]
          [
            set income draw-income
          ]
          set income-group find-income-group
        ]
      ]
    )
    ;; keep distro of cars wanting to park in model constant

    ask cars with [wants-to-park = 0] [set wants-to-park false]
    (ifelse (count cars with [wants-to-park] * 100 / count cars) > parking-cars-percentage
      [
        set park parking-cars-percentage +  random (100 - parking-cars-percentage)
      ]
      [
        set park random parking-cars-percentage
      ]
    )
  ]

  set wants-to-park false
  if park <= parking-cars-percentage [set wants-to-park true]

  set direction-turtle [direction] of patch-here
  set looks-for-parking? false
  ;; if placed on intersections, decide orientation randomly
  if intersection?
  [
    ifelse random 2 = 0
    [ set direction-turtle [dirx] of patch-here ]
    [ set direction-turtle [diry] of patch-here ]
  ]


  ;;
  set heading (ifelse-value
    direction-turtle = "up" [ 0 ]
    direction-turtle = "down"[ 180 ]
    direction-turtle = "left" [ 270 ]
    direction-turtle = "right"[ 90 ])

  ;; set goals for navigation
  ;set-navgoal
  ;set nav-prklist navigate patch-here nav-goal
  ;set nav-hastarget? false



  set park-time draw-park-duration
  set parked? false
  set reinitialize? true
  set die? false
  set wtp draw-wtp
  set wtp-increased 0

  ;; designate parking offenders (right now 25%)
  let offender-prob random 100
  ifelse offender-prob >= 75  [
    set parking-offender? true
  ]
  [
    set parking-offender? false
  ]

  set lots-checked no-patches

  ;; variables for utility function
  set distance-parking-target -99
  set price-paid -99
  set expected-fine -99
  set outcome -99
  set temp-utility-value -99

  ;; new code for parking behvavior
  ;let current-node one-of nodes-here
  ;let furthest-space max-one-of park-spaces [get-path-length (current-node) (get-closest-node lot-id)]
  ;set max-dist-parking-target get-path-length (current-node) (get-closest-node [lot-id] of furthest-space)
  let section [road-section] of patch-here
  set max-dist-parking-target table:get (table:get road-space-dict section) "max-dist-parking-target"

  ;; set goals for navigation
  set-navgoal
  let goal nav-goal

  set max-dist-parking-target distance max-one-of lots [distance goal]
  ; draw random purpose
  set purpose random 4

  ifelse use-synthetic-population [
    set logit-weights report-logit-weights
  ]
  [
    ;; axhausen: setting up default strategy flag value i.e. no switch when the agent is initiated
    set switch-strategy-flag 0
    ;; polak: initializing variables for informed and uninformed agents that are selecting specific strategies
    set informed-flag random 3 ;; polak: 0 denotes uninformed agent strategy, 1 informed agent strategy
    set agent-strategy-flag 0 ;; polak: default initialization, denoting no strategy, weights initialized randomly default value
    ifelse informed-flag = 1 or informed-flag = 2 [
      set agent-strategy-flag draw-informed-strategy-value
    ]
    [
      set agent-strategy-flag draw-uninformed-strategy-value
    ]
    ;; polak: below weights are stub values in-case 'agent-strategy-flag' is required to be set to 0
    set hard-weight-list n-values 5 [random 2] ;; polak: randomly setting default weights
    set fuzzy-weight-list map [i -> ifelse-value (i > 0)  [random-float i][i]] hard-weight-list ;; polak: setting random float weights based on binary weights
                                                                                                ;; polak: setting the weight values from the 'binary-weight-list' and 'fuzzy-weight-list' based on the 'agent-strategy-flag' values
    set hard-weight-list draw-hard-weights agent-strategy-flag hard-weight-list ;; two agruments for 'draw-binary-weights' function
    set fuzzy-weight-list draw-fuzzy-weights hard-weight-list ;; one agruments for 'draw-fuzzy-weights' function
  ]


  ;; set parking lot target according to utility function
  ifelse wants-to-park [

    set nav-prklist find-favorite-space      ;; polak: parsing 'fuzzy-weight-list' values to the 'navigate' function
  ]                                        ;if empty? nav-prklist [die]                                         ;; agent dies when already checked all parking spots
  [                                         ;set lot-ids-checked insert-item 0 lot-ids-checked first nav-prklist ;; insert lot-id of current parking target in checked lot-ids list to keep track on which lots have already been visited
    set nav-prklist []
  ]
  set nav-hastarget? false

end

;; Setup cars before starting simulation so as to hit the target occupancy (if possible)
to setup-parked
  ;foreach lot-colors [ lot-color ->
  ;let current-lot lots with [pcolor = lot-color]
  ;let occupancy (count cars-on current-lot / count current-lot)
  ;if occupancy < target-start-occupancy [
  if count cars-on park-spaces / count park-spaces < target-start-occupancy [
    while [not parked? and not empty? fav-lot-ids][
      let current-space item 0 fav-lot-ids
      let initial-lot one-of park-spaces with [lot-id = current-space  and not any? cars-on self]
      ifelse initial-lot != nobody [
        initial-park initial-lot
        if not parked? [
          set fav-lot-ids remove-item 0 fav-lot-ids
        ]
      ]
      [
        let current-lot park-spaces with [lot-id =  current-space]
        set lots-checked (patch-set lots-checked current-lot)
        set fav-lot-ids remove-item 0 fav-lot-ids
      ]
    ]
  ]
end

;; parks cars before start of simulation
to initial-park [initial-lot]
  ;; different routings for garages and roadside spaces
  ifelse member? initial-lot garages [
    let current-space initial-lot
    let parking-fee  [fee] of current-space  ;; compute fee
    ifelse use-synthetic-population or (wtp >= parking-fee)
    [
      move-to current-space
      ask current-space [set car? true]
      set paid? true
      set price-paid parking-fee
      set city-income city-income + parking-fee
      set parked? true
      set looks-for-parking? false
      set nav-prklist []
      set nav-hastarget? false
      set fee-income-share (parking-fee / (income / 12))
      set lots-checked no-patches
      set distance-parking-target distance nav-goal ;; update distance to goal (problematic here?)
      let gateway-x one-of [pxcor] of gateways with [lot-id = [lot-id] of current-space]
      let gateway-y one-of [pycor] of gateways with [lot-id = [lot-id] of current-space]
      set utility-value compute-utility current-space 0
      set outcome utility-value
      let current-lot park-spaces with [lot-id =   [lot-id] of current-space]
      set lots-checked (patch-set lots-checked current-lot)
      (foreach [0 0 1 -1] [-1 1 0 0] [[a b]->
        if ((member? patch (gateway-x + a) (gateway-y + b) roads))[
          set direction-turtle [direction] of patch (gateway-x + a) (gateway-y + b)
          set heading (ifelse-value
            direction-turtle = "up" [ 0 ]
            direction-turtle = "down"[ 180 ]
            direction-turtle = "left" [ 270 ]
            direction-turtle = "right"[ 90 ])
          stop
        ]
        ]
      )
      stop
    ]
    [
      let current-lot park-spaces with [lot-id =   [lot-id] of current-space]
      set lots-checked (patch-set lots-checked current-lot)
      stop
    ]
  ]
  [
    move-to initial-lot
    ask initial-lot [set car? true]
    set parked? true
    set looks-for-parking? false
    set nav-prklist []
    set nav-hastarget? false
    let parking-fee ([fee] of initial-lot )  ;; compute fee
    set fee-income-share (parking-fee / (income / 12))
    let current-lot park-spaces with [lot-id =   [lot-id] of initial-lot ]
    set lots-checked (patch-set lots-checked current-lot)
    ifelse (parking-offender?)
    [
      set paid? false
      set price-paid 0
    ]
    [
      set paid? true
      set price-paid parking-fee
    ]
    set-car-color
    set utility-value compute-utility initial-lot 0
    set outcome utility-value
    set distance-parking-target distance nav-goal ;; update distance to goal (problematic here?)
    (foreach [0 0 1 -1] [-1 1 0 0] [[a b]->
      if ((member? patch-at a b roads))[
        set direction-turtle [direction] of patch-at a b
        set heading (ifelse-value
          direction-turtle = "up" [ 0 ]
          direction-turtle = "down"[ 180 ]
          direction-turtle = "left" [ 270 ]
          direction-turtle = "right"[ 90 ])
        stop
      ]
      ]
    )
    stop
  ]
end

;; Find a road patch without any turtles on it and place the turtle there.
to put-on-empty-road  ;; turtle procedure
  move-to one-of initial-spawnpatches with [not any? cars-on self]
end

;; Compute the price you expect to pay for a parking lot
to-report compute-price [parking-lot]
  let expected-price 0
  let fine-probability compute-fine-prob park-time

  let parking-fee 0
  ;; account for group pricing
  ifelse group-pricing and not member? parking-lot garages
  [set parking-fee item income-group [group-fees] of parking-lot]
  [set parking-fee [fee] of parking-lot]

  ;; you have to pay in garages
  ifelse (parking-offender? and not member? parking-lot garages)[; and (wtp >= (parking-fee * fines-multiplier) * fine-probability))[
    set expected-price (parking-fee * fines-multiplier) * fine-probability
  ]
  [
    set expected-price parking-fee
  ]
  report expected-price
end

;; Compute the utility of a possible parking lot
to-report compute-utility [parking-lot count-passed-spots] ;; polak: parsing the 'fuzzy-weight-list' weight vector
  let goal nav-goal
  set distance-parking-target [distance goal] of parking-lot
  ;let distance-location-parking distance parking-lot ;; for this parking lot would need to be a patch


  let traffic 0
  let path-length 0
  let lotID [lot-id] of parking-lot
  let section [road-section] of patch-here


  ; get path length and current traffic estimate from road-space-dict
  set path-length table:get (table:get (table:get road-space-dict section) lotID) "path-length"
  if ticks > 0 [
    set traffic table:get (table:get (table:get road-space-dict section) lotID) "traffic"
  ]
  ;; convert patches to minutes
  let access path-length / 8
  let egress distance-parking-target / 4
  let search traffic



  let price compute-price parking-lot
  ;; set waiting-time wt-tm    ;; placeholder: commented out based the discussion
  let service [service?] of parking-lot ;; currently service is 0.25 for garages, 0 others
                                        ;; compute utility function
                                        ;print traffic
                                        ;print path-length
  let garage 0
  if member? parking-lot garages [ set garage 1]

  let utility 0
  ifelse use-synthetic-population
  [
    set utility report-utility access search egress price garage traffic
    ;if length remove-duplicates [lot-id] of lots-checked > 0 [
      ;show temp-utility-value
    ;]
    if utility > temp-utility-value [
      set temp-utility-value utility
      set computed-access access
      set final-access search-time
      set final-egress egress
      ifelse garage = 1 [
        set final-type "garage"
      ]
      [
        set final-type "curb"
      ]
    ]
  ]
  [
    ;;compute global maxima for agent to be used in utility function
    let max-path-length max-dist-parking-target
    set path-length path-length / max-path-length
    if path-length > 1 [set path-length 1]
    (ifelse
      ;; axhausen: more flexible parking strategy selection after strategy swapping after high wait time values
      wait-time > 2.5 * (temporal-resolution / 60) and wait-time < 5 * (temporal-resolution / 60) [ ;; conditional values tuned, Axhausen et al's 'stated preference experiment'.
                                                                                                    ;; print "Statement 1 Triggered"
        set fuzzy-weight-list draw-fuzzy-weights [0 1 0 1 0]
        set switch-strategy-flag 1
        set agent-strategy-flag 5
        set one-strategy-change-counter one-strategy-change-counter + 1]
      wait-time > 5 * (temporal-resolution / 60)  [ ;; conditional values tuned, Axhausen et al. 'stated preference experiment'.
                                                    ;; print "Statement 2 Triggered"
        set fuzzy-weight-list draw-fuzzy-weights [0 1 0 0 0]
        set switch-strategy-flag 2
        set switch-strategy-flag 8
        set two-strategy-change-counter two-strategy-change-counter + 1] ;; Flexible strategy, only concerned with distance location parking.
    )

    ;;initiate weights
    ;; let weight-list n-values 5 [random-float 1] ;; polak: added the fuzzy weight list for parking strategy influence
    ;; print fzy-wght-lst
    let weight-sum sum fuzzy-weight-list
    let norm-weight-list map [i -> i / weight-sum] fuzzy-weight-list ;; normalizes the weights such that they add up to 1
    let w1 item 0 norm-weight-list
    let w2 item 1 norm-weight-list
    let w3 item 2 norm-weight-list
    let w4 item 3 norm-weight-list
    let w5 item 4 norm-weight-list
    set utility (- (w1 * (distance-parking-target / max-dist-parking-target)) - (w2 * (0.5 * traffic + 0.5 * path-length)) - (w4 * (price / wtp)) + (w5 * service) + (count-passed-spots * 0.1) + random-normal 0 0.125)
  ]
  ;; - (w3 * (waiting-time / mean-wait-time))    ;; placeholder: commented out based the discussion

  report utility
end

to-report report-utility [access search egress price garage traffic]
  let access-w item 0 logit-weights
  let search-w item 1 logit-weights
  let egress-w item 2 logit-weights
  let fee-w item 3  logit-weights
  let type-w item 4 logit-weights
  let access-strategy-interaction-w item 5 logit-weights
  let search-strategy-interaction-w item 6 logit-weights
  let egress-strategy-interaction-w item 7 logit-weights
  let type-strategy-interaction-w item 8 logit-weights
  let fee-strategy-interaction-w item 9 logit-weights
  let access-purpose-interaction-w item 10 logit-weights
  let search-purpose-interaction-w item 11 logit-weights
  let egress-purpose-interaction-w item 12 logit-weights
  let type-purpose-interaction-w item 13 logit-weights
  let fee-purpose-interaction-w item 14 logit-weights
  let income-fee-interaction-w item 15 logit-weights
  ;let income-fee-interaction-6-w item 16 logit-weights
  let gender-w item 16 logit-weights
  (ifelse
    traffic > 0 and traffic < 0.12 [
      set access access * 1.1
    ]
    traffic > 0.12 and traffic < 0.25 [
      set access access * 1.25
    ]

    traffic > 0.25 [
      set access access * 1.5
    ]
  )
  let female 0
  if gender = 2 [set female 1] ; male is reference class (I know)
                               ; compute utility according to survey results (computed via multinomial logit model)
  let utility (access-w * access) + (access-strategy-interaction-w  * access) + (access-purpose-interaction-w  * access) +
  (search-w * search) + (search-strategy-interaction-w * search) + (search-purpose-interaction-w * search) +
  (egress-w * egress) + (egress-strategy-interaction-w * egress) + (egress-purpose-interaction-w * egress) +
  (type-w * garage) + (type-strategy-interaction-w * garage) + (type-purpose-interaction-w * garage) +
  (fee-w * price) + (fee-strategy-interaction-w * price) + (fee-purpose-interaction-w * price) + (income-fee-interaction-w * price) + ;((income-fee-interaction-6-w * income-interval-survey) ^ 6) +
  (gender-w * female)
  report utility
end

;; Determine parking space that maximizes utility
to-report find-favorite-space ;; polak: parsing the 'fuzzy-weight-list' weight vector
  let ut-list []
  ; in case helper value has already been written before
  set temp-utility-value -99
  let current patch-here
  ;;print "lot-ids-checked"
  ;;print lot-ids-checked

  if ((count lots-checked) > (count park-spaces / 2)) or (search-time > temporal-resolution) [
    set reinitialize? false
    set die? true
    set outcome min-util ;; cars will not find parking, get minimum utility
    report []
  ]
  ;if search-time > (temporal-resolution / 2) [report []]
  ;; for each patch in lots-list computes a temporary list including lot-id with respective utility
  ;; foreach ((range 1 (lot-counter))) [id ->
    foreach (remove-duplicates [lot-id] of lots) [id ->
    let lot one-of lots with [lot-id = id]

    ;; non-offenders should only consider spots they can afford
    let parking-fee 0
    ifelse group-pricing
    [set parking-fee item income-group [group-fees] of lot]
    [set parking-fee [fee] of lot]
    if not member? lot lots-checked[ ;and (parking-offender? or (parking-fee <= wtp)) [                       ;; only compute utilities for lot-ids which have not been checked before
      let utility compute-utility lot util-increase
      let tmp list id utility
      set ut-list lput tmp ut-list
    ]
  ]

  ;; for each patch in garages-list computes a temporary list including lot-id with respective utility
  ;; all utitilities and lot-ids are combined in ut-list
  foreach (range (lot-counter ) (lot-counter + num-garages)) [id ->
    let garage one-of garages with [lot-id = id]
    if not member? garage lots-checked and any? garages with [lot-id = id and car? != true][                       ;; only compute utilities for lot-ids which have not been checked before
      let utility compute-utility garage util-increase
      let tmp list id utility
      set ut-list lput tmp ut-list
    ]
  ]

  ;;print "ut-list"

  if empty? ut-list [
    set reinitialize? false
    set die? true
    set outcome min-util ;; cars will not find parking, get minimum utility
    report []
  ]
  let max-ut max map last ut-list

  ifelse not empty? ut-list and max-ut >= min-util [
    ;; now we need to group the utilities by lot-id and calculate the mean utility over the same lot-ids
    ;let grouped-list table:group-items ut-list [l -> first l] ;; all utilities grouped by their lot-id
    ;let lot-id-list table:keys grouped-list                   ;; list of all possible lot-ids being actual parking lots

    ;let mean-ut-list []
    ;foreach lot-id-list [id ->
    ;let all-ut-list map last table:get grouped-list id
    ;let mean-ut mean all-ut-list
    ;let tmp2 list id mean-ut
    ;set mean-ut-list lput tmp2 mean-ut-list                 ;; mean-ut-list gives us a nested list containing lot-id with its mean utility over all calculated utilities from the different patches
    ;]

    ;; compute maximum utility for each agent and save information together with respective lot-id in fav-lot
    set utility-value max-ut
    let fav-lot first filter [elem -> last elem = max-ut] ut-list      ;; favourite parking lot for the respective agent as a list of lot-id and utility
    let fav-lots-list sort-by [[list1 list2] -> last list1 > last list2] ut-list
    set fav-lot-ids map first fav-lots-list
    set fav-lot-id (list first fav-lot)                                     ;; lot-id where the agents wants to navigate to saved in a list
                                                                            ;; (for convenience such that we do not need to change other code since before it worked with a list)
  ]
  [
    set fav-lot-id []
    set fav-lot-ids []
    set reinitialize? false
    set die? true
    set outcome min-util ;; cars will not find parking, get minimum utility
  ]
  report fav-lot-id
end

;; Determine parking lots closest to current goal
to-report old-navigate [current goal]

  let fav-lots []
  let templots lots
  ;; check if there is any curbside space cheaper than garages and whether the garages are full, otherwise only check curbside parking
  if num-garages > 0[
    let garage-fee mean [fee] of garages
    if (not any? lots with [fee < garage-fee]) and ((count cars-on garages / count garages) < 1)[
      set templots (patch-set lots gateways)
    ]
  ]

  while [count templots > 0] [
    let i min-one-of templots [distance goal]
    set fav-lots insert-item 0 fav-lots [lot-id] of i
    set templots templots with [lot-id != [lot-id] of i]
    set color-counter color-counter + 1
    ;; check two streets per parking zone (otherwise cars search for too long)
    if color-counter = 2[
      set templots templots with [pcolor != [pcolor] of i]
      set color-counter 0
    ]
  ]
  set fav-lots reverse fav-lots
  set color-counter 0
  report fav-lots
end

;; assignment of navigation goal for new agents, spots in the center are more likely to become goals
to set-navgoal
  let max-distance max [center-distance] of potential-goals
  let switch random 100
  (ifelse
    switch <= 39 [
      set nav-goal one-of potential-goals with [center-distance <= max-distance * 0.35]
      if show-goals[
        ask one-of potential-goals with [center-distance <= max-distance * 0.35][
          set pcolor cyan
        ]
      ]
    ]
    switch > 39 and switch <= 65 [
      set nav-goal one-of potential-goals with [center-distance <= max-distance * 0.5 and center-distance > max-distance * 0.35]
      if show-goals[
        ask one-of potential-goals with [center-distance <= max-distance * 0.5 and center-distance > max-distance * 0.35][
          set pcolor pink
        ]
      ]
    ]
    switch > 65 and switch <= 80 [
      set nav-goal one-of potential-goals with [center-distance <= max-distance * 0.6 and center-distance > max-distance * 0.5]
      if show-goals[
        ask one-of potential-goals with [center-distance <= max-distance * 0.6 and center-distance > max-distance * 0.5][
          set pcolor violet
        ]
      ]
    ]
    switch > 80[
      set nav-goal one-of potential-goals with [center-distance <= max-distance and center-distance > max-distance * 0.6]
      if show-goals[
        ask one-of potential-goals with [center-distance <= max-distance and center-distance > max-distance * 0.6][
          set pcolor turquoise
        ]
      ]
  ])
end

;; draw turtle from synthetic population
to draw-synthetic-turtle

  let synthetic-turtle item 0 synthetic-population
  set gender item 1 synthetic-turtle
  set age item 2 synthetic-turtle
  set income item 3 synthetic-turtle
  set degree item 4 synthetic-turtle
  set school-degree item 5 synthetic-turtle
  set parking-strategy item 6 synthetic-turtle
  set income-interval-survey item 7 synthetic-turtle
end

;;;;;;;;;;;;;;;;;;;;;;;;
;; Runtime Procedures ;;
;;;;;;;;;;;;;;;;;;;;;;;;

;; Run the simulation, is called at every timestep
to go

  ;; have the intersections change their color
  set-signals
  set num-cars-stopped 0

  ;; set the turtles speed for this time thru the procedure, move them forward their speed,
  ;; record data for plotting, and set the color of the turtles to an appropriate color
  ;; based on their speed
  go-cars
  ;; update the phase and the global clock
  ;; control-lots
  ;; set prices dynamically
  ;; if dynamic-pricing-baseline [update-baseline-fees]
  update-demand-curve
  update-traffic-estimates
  if not debugging [recreate-cars]
  record-globals

  next-phase
  tick
end

;; sub-go-function for cars
to go-cars
  ask cars
  [
    ifelse not parked?
    [
      ;; check whether end of map is reached and turtle is supposed to die
      check-for-death
      ;if car has no target
      if not nav-hastarget?[
        ; set new target
        set-path
      ]

      ;; hotfix (should think of better solution)
      if nav-pathtofollow = false [
        show nav-pathtofollow
        show patch-here
        show search-time
        if reinitialize? [
          keep-distro income-group
          set cars-to-create cars-to-create +  1
        ]
        if document-turtles [document-turtle]
        if not reinitialize? [
          update-vanished
          ;set outcome min-util
        ]
        die
      ]

      ;==================================================
      ;decide whether to move forward
      navigate-road
      ;==================================================
      if looks-for-parking?;park <= parking-cars-percentage and  ;; x% of cars look for parking
      [
        park-car
        if member? patch-ahead 1 intersections [ ;;Dummy implementation
                                                 ;print self
                                                 ;print patch-here
                                                 ;print lots-checked
          set looks-for-parking? false
          set nav-prklist find-favorite-space
        ]
      ]
      record-data
      update-search-time
      ;;set-car-color
    ]
    [
      unpark-car
    ]
    ;;increments search-time if not parked
    ;show "final"
    ;show timer
  ]
end



;;;;;;;;;;;;;;;;
;; Navigation ;;
;;;;;;;;;;;;;;;;

;; check if turtle is leaving the grid (dying) and document appropriate statistics
to check-for-death
  if die? and (patch-ahead 1 = nobody or (member? patch-ahead 1 finalpatches)) [;; if, due to rounding, the car ends up on the final patch or the next patch is a final patch
    set traffic-counter traffic-counter + 1
    if reinitialize? [
      keep-distro income-group
      set cars-to-create cars-to-create +  1
    ]
    if document-turtles [document-turtle]
    if not reinitialize? [
      update-vanished
    ]
    die
  ]
end

;set path to follow
to set-path
  ;; if I have already parked, I can delete my parking list.

  let node-ahead one-of nodes-on patch-ahead 1
  ifelse not empty? nav-prklist
  ; set new path to first element of nav-prklist if not empty
  [
    ifelse node-ahead != nobody [
      set nav-pathtofollow determine-path node-ahead first nav-prklist
    ]
    [
      set nav-pathtofollow determine-path one-of nodes-on patch-here first nav-prklist
    ]
  ] ;; use patch-ahead because otherwise a node behind the car may be chosen, leading it to do a U-turn
    ;; if the parking list is empty either all parkingspots were tried or the car has already parked
  [
    ifelse node-ahead != nobody [
      set nav-pathtofollow determine-finaldestination node-ahead
    ]
    [
      set nav-pathtofollow determine-finaldestination one-of nodes-on patch-here
    ]
    set die? true
  ] ;; use patch-ahead because otherwise a node behind the car may be chosen, leading it to do a U-turn

  set nav-hastarget? true
end

; decide whether to move forward
to navigate-road
  ifelse not empty? nav-pathtofollow [
    if wait-time > temporal-resolution / 12 and member? patch-ahead 1 intersections[
      ;; alternative routing to avoid non-dissolvable congestion (after 5 Minutes of waiting)
      compute-alternative-route
    ]
    let nodex first nav-pathtofollow
    set-car-speed
    let x [xcor] of nodex
    let y [ycor] of nodex
    let patch-node patch x y
    face nodex ; might have to be changed
    set direction-turtle [direction] of patch-node
    fd speed
    ;once we reached node
    if one-of nodes-here = nodex [
      ;delete first node from nav-pathtofollow
      set nav-pathtofollow remove-item 0 nav-pathtofollow
    ]
    ;show "move"
    ;show timer
  ]
  [
    ;is looking for parking
    if wants-to-park and not empty? nav-prklist[
      if [road-section] of patch-here = get-road-section first nav-prklist[
        set looks-for-parking? true
      ]
    ]
    ;car currently has no target
    set nav-hastarget? false
    ; first item from prklist is deleted (has been  visited)
  ]
end

;; plot path to exit
to-report determine-finaldestination [start-node]
  let finalnodes nodes-on finalpatches
  let finalnode min-one-of finalnodes [length(nw:turtles-on-path-to one-of nodes-here)]
  let path 0
  ask start-node [set path nw:turtles-on-path-to finalnode]
  report path
end

;; plot path to parking street
to-report determine-path [start lotID]

  let node-proxy get-closest-node lotID

  let previous node-proxy
  ask node-proxy[
    let current one-of in-link-neighbors
    let next 0
    let indegree 1
    while [indegree = 1]
    [

      set previous current
      ask current [
        set next one-of in-link-neighbors
      ]
      set current next
      ask current [
        set indegree count in-link-neighbors
      ]
    ]
  ]
  let path 0
  let nodes-on-path 0
  ask start [
    set path nw:turtles-on-path-to previous
  ]

  report path
end


;; get clostest node to parking space
to-report get-closest-node [lotID]
  let lotproxy one-of original-lots with [lot-id = lotID]
  ;; if lot-id belongs to garage, navigate to gateway
  if num-garages > 0 [
    if any? gateways with [lot-id = lotID][
      set lotproxy gateways with [lot-id = lotID]
    ]
  ]
  let closest-node 0
  ask lotproxy [
    set closest-node one-of nodes-on neighbors4
  ]
  report closest-node
end

;; get id of closest road to lot
to-report get-road-section [lotID]
  let lotproxy one-of original-lots with [lot-id = lotID]
  ;; if lot-id belongs to garage, navigate to gateway
  if num-garages > 0 [
    if any? gateways with [lot-id = lotID][
      set lotproxy gateways with [lot-id = lotID]
    ]
  ]
  let road-patch 0
  ask lotproxy[
    set road-patch one-of neighbors4 with [member? self roads]
  ]
  report [road-section] of road-patch
end

;; compute path length between two nodes in the network
to-report get-path-length [startNode endNode]
  let path-length 0
  ask startNode [
    let nodes-on-path (turtle-set nw:turtles-on-path-to endNode)
    set path-length count (patch-set [patch-here] of nodes-on-path)
  ]
  report path-length
end

;; in cases of too much congestion, compute alternative route to destination
to compute-alternative-route
  ;; Check whether intersections lies at the outer border of the map
  let intersec patch-ahead 1
  let x-intersec [pxcor] of intersec
  let y-intersec [pycor] of intersec
  let cars-ahead 0

  ;; check what alternatives might be available
  let direct-x [dirx] of intersec
  let direct-y [diry] of intersec
  let x (ifelse-value
    direct-x = "left" [-1]
    direct-x = "right" [1]
  )
  let y (ifelse-value
    direct-y = "up" [1]
    direct-y = "down" [-1])


  let nodes-ahead one-of nodes-on patch-ahead 2
  let nodes-turn one-of nodes-on patch-at x y
  let path 0
  ifelse not member? nodes-ahead nav-pathtofollow [
    ask patch-ahead 2 [
      ;set pcolor turquoise
      set cars-ahead cars in-radius 1
    ]
    ifelse nodes-ahead != nobody and not any? cars-ahead[
      ;ask one-of nodes-on patch-at x y [set path nw:turtles-on-path-to nodes-ahead] ;;used to be intersec
      move-to patch-ahead 2
      ;ask one-of nodes-on intersec [set path nw:turtles-on-path-to nodes-ahead]
      set alternative? true
    ]
    [
      stop
    ]
  ]
  [
    ask patch-at x y [
      ;set pcolor magenta
      set cars-ahead cars in-radius 1
    ]
    ifelse nodes-turn != nobody and not any? cars-on patch-at x y[
      ;ask one-of nodes-on intersec [set path nw:turtles-on-path-to nodes-turn] ;;used to be intersec
      move-to patch-at x y
      set direction-turtle [direction] of patch-here
      set heading (ifelse-value
        direction-turtle = "up" [ 0 ]
        direction-turtle = "down"[ 180 ]
        direction-turtle = "left" [ 270 ]
        direction-turtle = "right"[ 90 ])
      set alternative? true
    ]
    [
      stop
    ]
  ]
  if alternative?[
    ifelse not empty? nav-prklist
    ; set new path to first element of nav-prklist if not empty
    [
      set nav-pathtofollow determine-path one-of nodes-on patch-ahead 1 first nav-prklist
    ] ;; use patch-ahead because otherwise a node behind the car may be chosen, leading it to do a U-turn
      ;; if the parking list is empty either all parkingspots were tried or the car has already parked
    [
      set nav-pathtofollow determine-finaldestination  one-of nodes-on patch-ahead 1
    ]
  ]
end

;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; Traffic Lights & Speed ;;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;


;; have the traffic lights change color if phase equals each intersections' my-phase
to set-signals
  if phase = 0 [
    ask intersections
    [
      set green-light-up? (not green-light-up?)
      set-signal-colors
    ]
  ]

  if phase >= ticks-per-cycle - ticks-per-cycle * 0.2[
    ask intersections
    [
      set-signal-yellow
    ]
  ]
end

;; This procedure checks the variable green-light-up? at each intersection and sets the
;; traffic lights to have the green light up or the green light to the left.
to set-signal-colors  ;; intersection (patch) procedure
  ifelse green-light-up?
  [
    if dirx = "right" and diry = "down"
    [
      ask patch-at -1 0 [ set pcolor green]
      ask patch-at 0 1 [ set pcolor red ]
    ]
    if dirx = "right" and diry = "up"
    [
      ask patch-at -1 0 [ set pcolor green]
      ask patch-at 0 -1 [ set pcolor red ]
    ]
    if dirx = "left" and diry = "down"
    [
      ask patch-at 1 0 [ set pcolor green]
      ask patch-at 0 1 [ set pcolor red ]
    ]
    if dirx = "left" and diry = "up"
    [
      ask patch-at 1 0 [ set pcolor green]
      ask patch-at 0 -1 [ set pcolor red]
    ]
  ]
  [
    if dirx = "right" and diry = "down"
    [
      ask patch-at -1 0 [ set pcolor red]
      ask patch-at 0 1 [ set pcolor green ]
    ]
    if dirx = "right" and diry = "up"
    [
      ask patch-at -1 0 [ set pcolor red]
      ask patch-at 0 -1 [ set pcolor green ]
    ]
    if dirx = "left" and diry = "down"
    [
      ask patch-at 1 0 [ set pcolor red]
      ask patch-at 0 1 [ set pcolor green ]
    ]
    if dirx = "left" and diry = "up"
    [
      ask patch-at 1 0 [ set pcolor red]
      ask patch-at 0 -1 [ set pcolor green ]
    ]
  ]
end

;; This procedure sets all traffic lights to yellow
to set-signal-yellow  ;; intersection (patch) procedure
  if dirx = "right" and diry = "down"
  [
    ask patch-at -1 0 [ set pcolor yellow + 1]
    ask patch-at 0 1 [ set pcolor yellow + 1 ]
  ]
  if dirx = "right" and diry = "up"
  [
    ask patch-at -1 0 [ set pcolor yellow + 1]
    ask patch-at 0 -1 [ set pcolor yellow + 1 ]
  ]
  if dirx = "left" and diry = "down"
  [
    ask patch-at 1 0 [ set pcolor yellow + 1]
    ask patch-at 0 1 [ set pcolor yellow + 1 ]
  ]
  if dirx = "left" and diry = "up"
  [
    ask patch-at 1 0 [ set pcolor yellow + 1]
    ask patch-at 0 -1 [ set pcolor yellow + 1]
  ]
end

;; set the turtles' speed based on whether they are at a red traffic light or the speed of the
;; turtle (if any) on the patch in front of them
to set-car-speed  ;; turtle procedure
  ifelse [pcolor] of patch-here = 15
  [ set speed 0 ]
  [ set-speed]
end

;; set the speed variable of the car to an appropriate value (not exceeding the
;; speed limit) based on whether there are cars on the patch in front of thecar
to set-speed  ;; turtle procedure
              ;; get the turtles on the patch in front of the turtle
  let cars-ahead other cars-on patch-ahead 1

  ;; if there are turtles in front of the turtle, slow down
  ;; otherwise, speed up
  ifelse any? cars-ahead
  [
    set speed [speed] of one-of cars-ahead
    slow-down
  ]
  [if [pcolor] of patch-here != red [speed-up]]

  ;;check for yellow lights
  if [pcolor] of patch-ahead 1 = yellow + 1 [
    slow-down
  ]
  ;; only drive on intersections if road afterwards is free
  if member? patch-ahead 1 intersections and is-list? nav-pathtofollow and length nav-pathtofollow  > 1[
    let node-after item 1 nav-pathtofollow
    let x [xcor] of node-after
    let y [ycor] of node-after
    let patch-after patch x y

    ask patch-after [
      set cars-ahead cars in-radius 1
    ]
    if any? (cars-ahead)[;any? cars-on patch-after or any? cars-on patch-after-after or  ;with [ direction-turtle != [direction-turtle] of myself ])[
      set speed 0
    ]
  ]
end

;; decrease the speed of the turtle
to slow-down  ;; turtle procedure
  ifelse speed <= 0  ;;if speed < 0
  [ set speed 0 ]
  [ set speed speed - acceleration ]
end

;; increase the speed of the turtle
to speed-up  ;; turtle procedure
  ifelse speed > speed-limit
  [ set speed speed-limit ]
  [ set speed speed + acceleration ]
end

;; set the color of the turtle to a different color based on whether the car is paying for parking
to set-car-color  ;; turtle procedure
  ifelse paid? != false
  [ set color grey]
  [ set color red]
end

;; keep track of the number of stopped turtles and the amount of time a turtle has been stopped
;; if its speed is 0
to record-data  ;; turtle procedure
  ifelse speed = 0
  [
    set num-cars-stopped num-cars-stopped + 1
    set wait-time wait-time + 1
  ]
  [ set wait-time 0 ]

end

to record-globals ;; keep track of all global reporter variables
  set mean-income mean [income] of cars
  set median-income median [income] of cars
  set n-cars count cars / num-cars
  set mean-wait-time mean [wait-time] of cars
  if count cars with [not parked?] > 0 [set mean-speed (mean [speed] of cars with [not parked?]) / speed-limit]

  set yellow-lot-current-fee ifelse-value count yellow-lot > 0 [mean [fee] of yellow-lot] [0]
  set green-lot-current-fee ifelse-value count green-lot > 0 [mean [fee] of green-lot] [0]
  set teal-lot-current-fee ifelse-value count teal-lot > 0 [mean [fee] of teal-lot] [0]
  set blue-lot-current-fee ifelse-value count blue-lot > 0 [mean [fee] of blue-lot] [0]

  set global-occupancy count cars-on park-spaces / count park-spaces

  set yellow-lot-current-occup ifelse-value count yellow-lot > 0 [count cars-on yellow-lot / count yellow-lot] [0]
  set green-lot-current-occup ifelse-value count green-lot > 0 [count cars-on green-lot / count green-lot] [0]
  set teal-lot-current-occup ifelse-value count teal-lot > 0 [count cars-on teal-lot / count teal-lot] [0]
  set blue-lot-current-occup ifelse-value count blue-lot > 0 [count cars-on blue-lot / count blue-lot] [0]
  if num-garages > 0 [set garages-current-occup count cars-on garages / count garages]
  set normalized-share-low ((count cars with [income-group = 0] / count cars)  / initial-low)
  if normalized-share-low > 1 [set normalized-share-low 1]

  if count cars with [not parked?] > 0 [set share-cruising count cars with [wants-to-park and not parked?] / count cars with [not parked?]]
  ;set income-entropy compute-income-entropy

  ;MO
  set allowed-parking-spaces [(list pxcor pycor ifelse-value allowed? [1] [0])] of park-spaces
end

;; cycles phase to the next appropriate value
to next-phase
  ;; The phase cycles from 0 to ticks-per-cycle, then starts over.
  set phase phase + 1
  if phase mod ticks-per-cycle = 0
    [ set phase 0 ]
end

;;;;;;;;;;;;;;;;;;;;;;;;
;; Parking procedures ;;
;;;;;;;;;;;;;;;;;;;;;;;;


;; handles parking behavior
to park-car ;;turtle procedure
            ;; check whether parking spot on left or right is available
  if (not parked? and (ticks > 0)) [
    (foreach [0 0 1 -1] [1 -1 0 0][ [a b] ->
      if [gateway?] of patch-at a b = true [
        park-in-garage patch-at a b
        set distance-parking-target distance nav-goal ;; update distance to goal
        stop
      ]
      if member? (patch-at a b) lots[
        let current-space patch-at a b
        ifelse not any? cars-on current-space[
          let parking-fee 0
          ifelse group-pricing
          [set parking-fee item income-group [group-fees] of current-space]
          [set parking-fee [fee] of current-space]  ;; compute fee
                                                   ;; check for parking offenders
          let fine-probability compute-fine-prob park-time
          ;; check if parking offender or WTP larger than fee
          if use-synthetic-population or wtp >= parking-fee or parking-offender? [
            ifelse (parking-offender?)[ ;and (wtp >= ([fee] of patch-at a b * fines-multiplier)* fine-probability ))[
              set paid? false
              set price-paid 0
              set expected-fine ([fee] of current-space * fines-multiplier)* fine-probability
              set city-loss city-loss + parking-fee
            ]
            [
              set paid? true
              set city-income city-income + parking-fee
              set price-paid parking-fee
              ;; keep track of checked lots
            ]
            set-car-color
            ;show [lot-id] of patch-at a b
            move-to current-space
            set parked? true
            set outcome utility-value
            set looks-for-parking? false
            set nav-prklist []
            set nav-hastarget? false
            set fee-income-share (parking-fee / (income / 12)) ;; share of monthly income
            ask patch-at a b [set car? true]
            let lot-identifier [lot-id] of current-space ;; value of lot-variable for current lot
            let current-lot lots with [lot-id = lot-identifier]
            set lots-checked (patch-set lots-checked current-lot)
            set distance-parking-target distance nav-goal ;; update distance to goal
            stop
          ]
        ]
        [
          if not member? current-space lots-checked
          [
            let lot-identifier [lot-id] of current-space ;; value of lot-variable for current lot
            let current-lot lots with [lot-id = lot-identifier]
            set lots-checked (patch-set lots-checked current-lot)
            update-wtp
          ]
          ;stop
        ]
      ]
      ]
    )
  ]
end

;; procedure to park in garage
to park-in-garage [gateway]
  let current-garage garages with [lot-id = [lot-id] of gateway]
  ifelse (count cars-on current-garage / count current-garage) < 1[
    let parking-fee (mean [fee] of current-garage)  ;; compute fee
    if use-synthetic-population or (wtp >= parking-fee)
    [
      let space one-of current-garage with [not any? cars-on self]
      move-to space
      ask space [set car? true]
      set paid? true
      set price-paid parking-fee
      set city-income city-income + parking-fee
      set parked? true
      set outcome utility-value
      set looks-for-parking? false
      set nav-prklist []
      set nav-hastarget? false
      set fee-income-share (parking-fee / (income / 12))
      ;set lots-checked no-patches
      let lot-identifier [lot-id] of gateway ;; value of lot-variable for current garage
      let current-lot park-spaces with [lot-id = lot-identifier]
      set lots-checked (patch-set lots-checked current-lot)
      stop
    ]
  ]
  [
    if not member? gateway lots-checked
    [
      let lot-identifier [lot-id] of gateway ;; value of lot-variable for current garage
      let current-lot park-spaces with [lot-id = lot-identifier]
      set lots-checked (patch-set lots-checked current-lot)
      update-wtp
    ]
    stop
  ]
end

;; handles unparking
to unpark-car ;; turtle procedure
  ifelse (time-parked < park-time)[
    set time-parked time-parked + 1
  ]
  [
    if num-garages > 0 and member? patch-here garages [
      unpark-from-garage
      stop
    ]
    (foreach [0 0 1 -1] [-1 1 0 0] [[a b]->
      if ((member? patch-at a b roads) and (not any? cars-at a b))[
        set direction-turtle [direction] of patch-at a b
        set heading (ifelse-value
          direction-turtle = "up" [ 0 ]
          direction-turtle = "down"[ 180 ]
          direction-turtle = "left" [ 270 ]
          direction-turtle = "right"[ 90 ])
        move-to patch-at a b
        set parked? false
        set time-parked 0
        set-car-color
        set reinitialize? true
        ask patch-here[
          set car? false
        ]
        stop
      ]
    ])
  ]
end

;; unparking from garage
to unpark-from-garage ;;
  let space patch-here
  let gateway gateways with [lot-id = [lot-id] of space]
  let road []
  ask gateway [set road one-of neighbors4 with [member? self roads]] ;; must use one-of to interpret as single agent
  if not any? cars-on road [
    set direction-turtle [direction] of road
    move-to road
    set parked? false
    ;set park 100
    set time-parked 0
    set-car-color
    set reinitialize? true
    ask space[
      set car? false
    ]
    stop
  ]
end

;; document turtle in csv
to document-turtle;;
  ifelse search-time > final-access [
    set final-access ((search-time - final-access) / temporal-resolution) * 60
  ]
  [
    ifelse search-time = 0 [
      set final-access computed-access
    ]
    [
      set final-access ((search-time) / temporal-resolution) * 60
    ]
  ]
  let converted-search-time (search-time / temporal-resolution) * 60
  let garage 0
  if final-type = "garage" [set garage 1]
  let checked-blocks length remove-duplicates [lot-id] of lots-checked
  let final-outcome 0
  ifelse wants-to-park and reinitialize? [
    set final-outcome report-utility final-access converted-search-time final-egress price-paid garage 0
  ]
  [
    set final-outcome outcome
  ]


  file-print  csv:to-row [(list who income income-group income-interval-survey age gender school-degree degree wtp parking-strategy purpose parking-offender? checked-blocks final-access final-egress final-type price-paid converted-search-time wants-to-park die? reinitialize? final-outcome)] of self
end

;; document traffic in csv
to document-traffic
    file-open output-traffic-file-path
    foreach road-sections [ rs ->
        let cars-on-road-section cars-on roads with [road-section = rs]
        if count cars-on-road-section > 0 [
            file-print csv:to-row (list ticks rs (count cars-on-road-section) (mean [speed] of cars-on-road-section))
        ]
    ]
end

;; report traffic on all road sections
to-report report-traffic
    report map report-road-section-traffic road-sections
end

to-report report-road-section-traffic [rs]
    let cars-on-road-section cars-on roads with [road-section = rs]
    report (list rs count cars-on-road-section ifelse-value count cars-on-road-section > 0 [mean [speed] of cars-on-road-section] [0])
end


;; compute utility (currently deprecated)
to compute-outcome
  ;; consider only cars trying to park
  let parking-cars cars with [park <= parking-cars-percentage ]
  if count parking-cars > 0 [
    let access-factor 20 / 1800 ;; 20$ per hour
    let egress-factor 50 / (60 * 60) ;; 50$ per hour, translated to seconds
    ask parking-cars [
      set outcome wtp
      set outcome outcome - access-factor * search-time
      if price-paid != -99 [set outcome outcome - price-paid] ;; check whether any price was paid
      if distance-parking-target != -99 [set outcome outcome - distance-parking-target * egress-factor * (5 / 1.4)] ;; 5 patches  = 1 meter, 1.4 meter per second
      if expected-fine != -99 [set outcome outcome - expected-fine]
    ]
    let min-outcome min [outcome] of cars with [outcome != -99]
    ;let max-outcome max [outcome] of cars with [outcome != -99]
    ;let outcome-range max-outcome - min-outcome
    ask cars with [outcome != -99] [
      ;set outcome (outcome - min-outcome) / outcome-range
      set outcome outcome + abs min-outcome
    ]
  ]
end

;; reporter to get group averages of outcomes
to-report get-outcomes [group]
  ifelse group = "all" [
    ifelse count cars with [outcome != -99] > 0 [
      report [outcome] of cars with [outcome != -99]
    ]
    [
      report 0
    ]
  ]
  [
    ifelse count cars with [outcome != -99 and income-group = group] > 0 [
      report [outcome] of cars with [outcome != -99 and income-group = group]
    ]
    [
      report 0
    ]
  ]
end

;; compute gini coefficient of outcomes
to-report compute-gini
  let sorted-outcome sort [outcome] of cars with [outcome != -99]
  let height 0
  let area 0
  foreach sorted-outcome [oc ->
    set height height + oc
    set area area + (height - oc / 2)
  ]
  let fair-area height * (length sorted-outcome / 2 )
  report (fair-area - area) / fair-area
end


;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; Environment procedures ;;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;

;; handles dynamic baseline pricing (modeled after SFpark)
to update-baseline-fees;;
  if (ticks mod (temporal-resolution / 2) = 0 and ticks > 0) [ ;; update fees every half hour
    let occupancy 0
    foreach lot-colors [ lot-color ->
      let current-lot lots with [pcolor = lot-color]
      if any? cars-on current-lot [
        set occupancy (count cars-on current-lot / count current-lot)
      ]


      (ifelse
        occupancy >= 0.9 [
          change-fee current-lot 0.25
        ]
        occupancy < 0.75 and occupancy >= 0.3 [
          change-fee current-lot -0.25
        ]
        occupancy < 0.3 and mean [fee] of current-lot >= 1 [
          change-fee current-lot -0.5
        ]
      )
    ]
  ]
end

;; update demand to reflect changes over the course of a typical business day in Mannheim (based on data on the utilised capacity of parking garages)
to update-demand-curve
  if (ticks mod (temporal-resolution / 2) = 0) [ ;; update demand every half hour
    let x ticks / temporal-resolution + 8
    set parking-cars-percentage ((-5.58662028e-04 * x ^ 3 + 2.76514862e-02 * x ^ 2 + -4.09343614e-01 *  x +  2.31844786e+00)  + demand-curve-intercept) * 100
  ]
end

;; update road-space-dict with current data on traffic
to update-traffic-estimates
  if (ticks mod (temporal-resolution / 2) = 0) [ ;; update demand every half hour
    foreach (remove-duplicates [road-section] of roads)
    [ section ->
      foreach (remove-duplicates [lot-id] of park-spaces)  [space ->
        ;let lot one-of lots with [lot-id = id]
        let patches-on-path table:get (table:get (table:get road-space-dict section) space) "patches"
        table:put (table:get (table:get road-space-dict section) space) "traffic" (count cars-on patches-on-path / (count  patches-on-path))
      ]
    ]
  ]
end

;; for changing prices during Reinforcement Learning
to change-fee [lot fee-change]
  let new-fee (mean [fee] of lot) + fee-change
  if new-fee < 0 [stop]
  ask lot [set fee fee + fee-change]
end

;; for changing prices of income groups
to change-group-fees [lot fee-change]
  ask lot [set group-fees (map [[a b] -> a + b] group-fees fee-change)]
end

to change-group-fees-free [lot new-fees]
  ask lot [set group-fees new-fees]
end

;; for free price setting of RL agent
to change-fee-free [lot new-fee]
  ask lot [set fee new-fee]
end

;; increase WTP for cars still searching
to update-wtp ;;
              ;; cars that did not find a place do not respawn
  if empty? nav-prklist [
    set reinitialize? false
    set die? true
    ; assign minimum utility for displaced cars
    set outcome min-util
  ]

  if wtp-increased <= 5 and not use-synthetic-population
    [
      set wtp wtp + wtp * .05
      set wtp-increased wtp-increased + 1
  ]
end

;; create new cars to replace those that left the simulation after parking (or without wanting to park)
to recreate-cars;;
  create-cars cars-to-create
  [
    set reinitialize? true
    setup-cars
    set-car-color
    record-data
    if not wants-to-park [
      set nav-prklist []
      set reinitialize? true
    ]
  ]
  set cars-to-create 0
end

;; keep distribution of incomes approx. constant
to keep-distro [income-class]
  (ifelse
    income-class = 0 [
      set low-to-create low-to-create + 1
    ]
    income-class = 1 [
      set middle-to-create middle-to-create + 1
    ]
    income-class = 2 [
      set high-to-create high-to-create + 1
    ]
  )
end

;; update search-time only for those wanting to park right now
to update-search-time
  if not parked? and wants-to-park and not empty? nav-prklist
  [set search-time search-time + 1]
end


;; randomly check for parking offenders every hour
to control-lots
  if ticks > 0 and (ticks mod (temporal-resolution / controls-per-hour) = 0) [
    let switch random 4
    (ifelse
      switch = 0 [
        let potential-offenders cars-on yellow-lot
        let fines (count potential-offenders with [not paid?]) *  fines-multiplier * mean [fee] of yellow-lot
        set city-income city-income + fines
        set total-fines total-fines + fines
      ]
      switch = 1 [
        let potential-offenders cars-on teal-lot
        let fines (count potential-offenders with [not paid?]) * fines-multiplier * mean [fee] of teal-lot
        set city-income city-income + fines
        set total-fines total-fines + fines
      ]
      switch = 2 [
        let potential-offenders cars-on green-lot
        let fines (count potential-offenders with [not paid?]) * fines-multiplier * mean [fee] of green-lot
        set city-income city-income + fines
        set total-fines total-fines + fines
      ]
      switch = 3[
        let potential-offenders cars-on blue-lot
        let fines (count potential-offenders with [not paid?]) * fines-multiplier * mean [fee] of blue-lot
        set city-income city-income + fines
        set total-fines total-fines + fines
    ])
  ]
end

;;computes probabilty to get caught for parking-offenders
to-report compute-fine-prob [parking-time]
  let n-controls round(parking-time / (temporal-resolution / controls-per-hour))
  ifelse n-controls <= 1 [
    report 0.25
  ]
  [
    let prob 0.25
    while [n-controls > 1][
      set prob prob + (0.75 ^ (n-controls - 1) * 0.25)
      set n-controls n-controls - 1
    ]
    report prob
  ]
end


;;;;;;;;;;;;;;;;;;;;;;
;; Income Reporter ;;
;;;;;;;;;;;;;;;;;;;;;;

;; draw parking duration following a gamma distribution
to-report draw-park-duration
  let minute temporal-resolution / 60
  let shift temporal-resolution / 3 ;; have minimum of 20 minutes
  set shift 0
  let mu 227.2 * minute
  let sigma (180 * minute) ^ 2
  report random-gamma ((mu ^ 2) / sigma) (1 / (sigma / mu)) + shift
end

;; global reporter: draws a random income, based on the distribution provided by the user
to-report draw-income
  let sigma  sqrt (2 * ln (pop-mean-income / pop-median-income))
  let mu     (ln pop-median-income)
  report exp random-normal mu sigma
end

;;global reporter, draws a random income based on the distribution in the sample
to-report draw-sampled-income
  ;; use absolute value for cases in which median becomes larger than mean (not in use currently)
  let sigma  sqrt abs (2 * ln (mean-income / median-income))
  let mu     (ln median-income)
  report exp random-normal mu sigma
end


;; designate income groups following the OECD standard
to-report find-income-group

  (ifelse
    income > (pop-median-income * 2)
    [
      report 2
    ]
    income < (pop-median-income * 0.75)
    [
      report 0
    ]
    [
      report 1
    ]
  )
end

;; draw wtp for different income groups from a gamma distribution (not used if synthetic population is used)
to-report draw-wtp ;;
  let mu 0
  let sigma 0
  (ifelse
    income-group = 0 [
      set mu 2.5
      set sigma mu * 0.25
    ]
    income-group = 1 [
      set mu 4.5
      set sigma mu * 0.30
    ]
    income-group = 2 [
      set mu 8
      set sigma mu * 0.45
    ]
  )
  ;;report abs (random-normal mu sigma)
  report random-gamma ((mu ^ 2) / sigma) (1 / (sigma / mu))
end

;; compute entropy of incomes
to-report compute-income-entropy
  let prop-low (count cars with [income-group = 0] / count cars)
  let prop-middle (count cars with [income-group = 1] / count cars)
  let prop-high (count cars with [income-group = 2] / count cars)
  let entropy 0
  foreach (list prop-low prop-middle prop-high) [ class ->
    if class > 0 [
      set entropy entropy + class * ln class
    ]
  ]

  let max-entropy -3 * (1 / 3 * ln (1 / 3))
  report (- entropy / max-entropy)
end

;; update counter for cars that did not find parking and will thus not respawn
to update-vanished
  (ifelse
    income-group = 0 [
      set vanished-cars-low vanished-cars-low + 1
    ]
    income-group = 1 [
      set vanished-cars-middle vanished-cars-middle + 1
    ]
    income-group = 2 [
      set vanished-cars-high vanished-cars-high + 1
    ]
  )
end

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; Parking Strategy Utilities  ;;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;


;; polak: draw parking strategy distribution based on Polak et al., Parking Search Behaviour
to-report draw-informed-strategy-value
  let n-birmingham 147
  let n-kingston 624

  let same-park-value ( n-birmingham * 39 + n-kingston * 33 ) / (n-birmingham + n-kingston)
  let private-park-value ( n-birmingham * 3 + n-kingston * 16 ) / (n-birmingham + n-kingston)
  let destination-reach-park-value ( n-birmingham * 18 + n-kingston * 18 ) / (n-birmingham + n-kingston)
  let nearest-goal-park-value ( n-birmingham * 26 + n-kingston * 18 ) / (n-birmingham + n-kingston)
  let active-lookup-park-value ( n-birmingham * 11 + n-kingston * 8 ) / (n-birmingham + n-kingston)

  let total-value-bound ( same-park-value + private-park-value + destination-reach-park-value + nearest-goal-park-value + active-lookup-park-value )
  let switch-value random (total-value-bound + 1)
  let informed-report-value 5
  (ifelse
    switch-value <= same-park-value [ set informed-report-value 1 ]
    switch-value > same-park-value and switch-value <= (same-park-value + private-park-value) [ set informed-report-value 2 ]
    switch-value > (same-park-value + private-park-value) and switch-value <= (same-park-value + private-park-value + destination-reach-park-value) [ set informed-report-value 3 ]
    switch-value > (same-park-value + private-park-value + destination-reach-park-value) and switch-value <= (same-park-value + private-park-value + destination-reach-park-value + nearest-goal-park-value) [ set informed-report-value 4 ]
    switch-value > (same-park-value + private-park-value + destination-reach-park-value + nearest-goal-park-value) and switch-value <= (same-park-value + private-park-value + destination-reach-park-value + nearest-goal-park-value + active-lookup-park-value) [ set informed-report-value 5 ]
  )
  report informed-report-value
end

;; polak: draw parking strategy distribution based on Polak et al., Parking Search Behaviour
to-report draw-uninformed-strategy-value
  let n-birmingham 147
  let n-kingston 624

  ;; let same-park-value ( n-birmingham * 39 + n-kingston * 33 ) / (n-birmingham + n-kingston)
  ;; let private-park-value ( n-birmingham * 3 + n-kingston * 16 ) / (n-birmingham + n-kingston)
  let destination-reach-park-value ( n-birmingham * 18 + n-kingston * 18 ) / (n-birmingham + n-kingston)
  ;; let nearest-goal-park-value ( n-birmingham * 26 + n-kingston * 18 ) / (n-birmingham + n-kingston)
  let active-lookup-park-value ( n-birmingham * 11 + n-kingston * 8 ) / (n-birmingham + n-kingston)

  ;; let total-value-bound ( same-park-value + private-park-value + destination-reach-park-value + nearest-goal-park-value + active-lookup-park-value )
  let total-value-bound ( destination-reach-park-value + active-lookup-park-value )
  let switch-value random (total-value-bound + 1)
  let uninformed-report-value 6
  (ifelse
    switch-value <= destination-reach-park-value [ set uninformed-report-value 6 ]
    switch-value > destination-reach-park-value and switch-value <= ( destination-reach-park-value + active-lookup-park-value ) [ set uninformed-report-value 7 ]
  )
  report uninformed-report-value
end

;; polak: setting binary weight values based on strategy values, Polak et al., Parking Search Behaviour
to-report draw-hard-weights [ag-strat-flg hrd-wghts]
  let new-hrd-wghts hrd-wghts
  (ifelse
    ;; informed strategies values
    ag-strat-flg = 1 [ set new-hrd-wghts [1 1 0 0 1] ]
    ag-strat-flg = 2 [ set new-hrd-wghts [1 0 1 0 1] ]
    ag-strat-flg = 3 [ set new-hrd-wghts [1 1 0 1 0] ]
    ag-strat-flg = 4 [ set new-hrd-wghts [1 0 1 0 0] ]
    ag-strat-flg = 5 [ set new-hrd-wghts [0 1 0 1 0] ]
    ;; uninformed strategies values, Chaniotakis et al.; Parmar et al.
    ag-strat-flg = 6 [ set new-hrd-wghts [1 0 1 1 0.25] ]
    ag-strat-flg = 7 [ set new-hrd-wghts [0 0 1 1 0.25] ]
  )
  report new-hrd-wghts
end

;; polak: converting hard weights to fuzzy weights, adding normalized distribution for weight selection
to-report draw-fuzzy-weights [hrd-wghts]
  let weight-noise random-normal 0.0 0.125
  let new-fuz-wghts map [i -> ifelse-value (i = 1)  [i - 0.3 + weight-noise][i + 0.3 + weight-noise]] hrd-wghts
  report new-fuz-wghts
end

to-report report-logit-weights []
  let weight-vector []
  set weight-vector lput -0.0893748 weight-vector ;; access
  set weight-vector lput 0 weight-vector ;; search
  set weight-vector lput (random-normal -0.2329862 0.2188632)  weight-vector ;; egress
  set weight-vector lput (random-normal -1.0354294 0.7221745)  weight-vector ;; fee
  set weight-vector lput 0 weight-vector ;; type-garage
  let access-strategy-interaction-weight 0
  let search-strategy-interaction-weight 0
  let egress-strategy-interaction-weight 0
  let type-strategy-interaction-weight 0
  let fee-strategy-interaction-weight 0
  (ifelse
    parking-strategy = 4 [
      ;set search-strategy-interaction-weight random-normal -0.087 0.2
      set type-strategy-interaction-weight random-normal 1.0272934 1.0473519
      set fee-strategy-interaction-weight 0.4833433
    ]
    parking-strategy = 5 [
      set egress-strategy-interaction-weight 0.1047587
    ]
    parking-strategy = 7[
      set egress-strategy-interaction-weight 0.1459745
      ;set search-strategy-interaction-weight -0.0887736
    ]
  )
  set weight-vector lput access-strategy-interaction-weight weight-vector
  set weight-vector lput search-strategy-interaction-weight weight-vector
  set weight-vector lput egress-strategy-interaction-weight weight-vector
  set weight-vector lput type-strategy-interaction-weight weight-vector
  set weight-vector lput fee-strategy-interaction-weight weight-vector

  let access-purpose-interaction-weight 0
  let search-purpose-interaction-weight 0
  let egress-purpose-interaction-weight 0
  let type-purpose-interaction-weight 0
  let fee-purpose-interaction-weight 0
  (ifelse
    purpose = 1 [ ;doctor
      set egress-purpose-interaction-weight -0.0815141
      set fee-purpose-interaction-weight 0.7507345
    ]
    purpose = 2 [ ;meeting friends
      set type-purpose-interaction-weight random-normal 0.3900153 0.0128061
      set fee-purpose-interaction-weight random-normal 0.2571034  0.4370313

    ]
    purpose = 3 [ ;shopping
      set egress-purpose-interaction-weight random-normal  -0.1690142  0.163935
      set fee-purpose-interaction-weight random-normal 0.4643296 0.5836606
    ]
  )
  set weight-vector lput access-purpose-interaction-weight weight-vector
  set weight-vector lput search-purpose-interaction-weight weight-vector
  set weight-vector lput egress-purpose-interaction-weight weight-vector
  set weight-vector lput type-purpose-interaction-weight weight-vector
  set weight-vector lput fee-purpose-interaction-weight weight-vector

  let income-fee-interaction 0
  (ifelse
    income-interval-survey = 2 [
      set income-fee-interaction -1.7680264
    ]
    income-interval-survey = 3 [
      set income-fee-interaction random-normal -1.1011304 0.9414337
    ]
    income-interval-survey = 4 [
      set income-fee-interaction random-normal -1.0112628  0.5049523
    ]
    income-interval-survey = 5 [
      set income-fee-interaction random-normal -0.9657823  0.6099223
    ]
    income-interval-survey = 6 [
      set income-fee-interaction random-normal -0.8719359  0.3795559
    ]
    income-interval-survey = 7  [
      set income-fee-interaction -0.7972391
    ]
  )

  set weight-vector lput income-fee-interaction weight-vector ;; income-fee-interaction
                                                              ;set weight-vector lput random-normal 0.1489018  0.8409924  weight-vector ;; ^6-income-fee-interaction
  set weight-vector lput 0.2414542 weight-vector ;; gender



  report weight-vector

end

;; MO
to update-parking []
  set yellow-lot (original-yellow-lot with [allowed?])
  set green-lot (original-green-lot with [allowed?])
  set teal-lot (original-teal-lot with [allowed?])
  set blue-lot (original-blue-lot with [allowed?])
  set lots (original-lots with [allowed?])
  set garages (original-garages with [allowed?])
  set park-spaces (original-park-spaces with [allowed?])
end

; Copyright 2003 Uri Wilensky.
; See Info tab for full copyright and license.
@#$#@#$#@
GRAPHICS-WINDOW
280
10
1438
1453
-1
-1
14.2
1
9
1
1
1
0
1
1
1
-40
40
-50
50
1
1
1
ticks
60.0

PLOT
2301
327
2734
642
Average Wait Time of Cars
Time
Average Wait
0.0
100.0
0.0
5.0
true
false
"" ""
PENS
"Waittime" 1.0 0 -16777216 true "" "plot mean-wait-time"
"Average speed" 1.0 0 -2674135 true "" "plot mean-speed"

SLIDER
5
45
270
78
num-cars
num-cars
10
1000
595.0
5
1
NIL
HORIZONTAL

PLOT
1448
15
1854
311
Share of Cars per Income Class
Time
%
0.0
21600.0
0.0
100.0
true
true
"" ""
PENS
"High Income" 1.0 0 -16777216 true "" "if count cars with [income-group = 2] > 0 [plot ((count cars with [income-group = 2] / count cars) * 100)]"
"Middle Income" 1.0 0 -13791810 true "" "if count cars with [income-group = 1] > 0 [plot ((count cars with [income-group = 1] / count cars) * 100)]"
"Low Income" 1.0 0 -2674135 true "" "ifelse count cars with [income-group = 0] != 0 [plot ((count cars with [income-group = 0] / count cars) * 100)][plot 0]"
"Share of intially spawned cars" 1.0 0 -7500403 true "" "plot (n-cars) * 100"
"Entropy" 1.0 0 -955883 true "" "plot income-entropy * 100"

BUTTON
140
10
271
43
Go
go
T
1
T
OBSERVER
NIL
NIL
NIL
NIL
0

BUTTON
5
10
138
43
Setup
setup
NIL
1
T
OBSERVER
NIL
NIL
NIL
NIL
1

SLIDER
5
80
140
113
ticks-per-cycle
ticks-per-cycle
1
100
20.0
1
1
NIL
HORIZONTAL

SLIDER
5
565
270
598
blue-lot-fee
blue-lot-fee
0
20
2.0
0.5
1
€ / hour
HORIZONTAL

SLIDER
5
465
270
498
yellow-lot-fee
yellow-lot-fee
0
20
2.5
0.5
1
€ / hour
HORIZONTAL

SLIDER
5
530
270
563
teal-lot-fee
teal-lot-fee
0
20
2.0
0.5
1
€ / hour
HORIZONTAL

SLIDER
5
495
270
528
green-lot-fee
green-lot-fee
0
20
2.0
0.5
1
€ / hour
HORIZONTAL

PLOT
1860
325
2289
640
Utilized Capacity at Different Lots
Time
Utilized Capacity in %
0.0
21600.0
0.0
100.0
true
true
"set-plot-background-color grey - 2\n" ""
PENS
"Blue Lot" 1.0 0 -13740902 true "" "plot blue-lot-current-occup * 100"
"Yellow Lot" 1.0 0 -855445 true "" "plot yellow-lot-current-occup * 100"
"Green Lot" 1.0 0 -8732573 true "" "plot green-lot-current-occup * 100"
"Teal  Lot" 1.0 0 -14520940 true "" "plot teal-lot-current-occup * 100"
"Garages" 1.0 0 -15520724 true "" "if num-garages > 0 [plot garages-current-occup * 100]"
"Overall Occupancy" 1.0 0 -7500403 true "" "plot global-occupancy * 100"
"Target Range" 1.0 2 -2674135 true "" "plot 75\nplot 90"

MONITOR
2115
760
2250
805
Mean Income in Model
mean [income] of cars
2
1
11

SLIDER
10
670
270
703
pop-median-income
pop-median-income
1000
40000
2956.0
1
1
€
HORIZONTAL

SLIDER
10
635
270
668
pop-mean-income
pop-mean-income
0
50000
3612.0
1
1
€
HORIZONTAL

TEXTBOX
110
440
214
462
Initial Fees
15
0.0
1

MONITOR
1947
830
2028
875
blue-lot-fee
mean [fee] of blue-lot
17
1
11

MONITOR
1942
675
2021
720
yellow-lot-fee
mean [fee] of yellow-lot
17
1
11

MONITOR
1944
781
2026
826
teal-lot-fee
mean [fee] of teal-lot
17
1
11

MONITOR
1942
729
2029
774
green-lot-fee
mean [fee] of green-lot
17
1
11

PLOT
1862
15
2292
311
Descriptive Income Statistics
Time
Euro
0.0
7200.0
0.0
5000.0
true
true
"" ""
PENS
"Mean" 1.0 0 -16777216 true "" "if count cars > 0 [plot mean-income]"
"Median" 1.0 0 -2674135 true "" "if count cars > 0 [plot median-income]"
"Standard Deviation" 1.0 0 -13791810 true "" "if count cars > 0 [plot standard-deviation [income] of cars ]"

PLOT
1450
1235
1855
1534
Average Search Time per Income Class
Time
Time
0.0
7200.0
0.0
1500.0
true
true
"" ""
PENS
"High Income" 1.0 0 -16777216 true "" "ifelse count cars with [income-group = 2] != 0 [plot mean [search-time] of cars with [income-group = 2 and park <= parking-cars-percentage]][plot 0] "
"Middle Income" 1.0 0 -13791810 true "" "if count cars with [income-group = 1] > 0 [plot mean [search-time] of cars with [income-group = 1 and park <= parking-cars-percentage]]"
"Low Income" 1.0 0 -2674135 true "" "ifelse count cars with [income-group = 0] != 0 [plot mean [search-time] of cars with [income-group = 0 and park <= parking-cars-percentage]][plot 0] "

TEXTBOX
65
605
256
630
Income Distribution
20
0.0
1

TEXTBOX
85
412
235
437
Parking Fees
20
0.0
1

SWITCH
5
140
140
173
hide-nodes
hide-nodes
0
1
-1000

SLIDER
5
250
270
283
lot-distribution-percentage
lot-distribution-percentage
0
1
0.55
0.05
1
NIL
HORIZONTAL

MONITOR
2115
710
2250
755
Min Income in Model
min [income] of cars
2
1
11

MONITOR
2115
814
2249
859
Max Income in Model
Max [income] of cars
2
1
11

SWITCH
140
80
270
113
show-goals
show-goals
1
1
-1000

PLOT
1860
955
2260
1230
Share of parked Cars per Income Class
Time
%
0.0
7200.0
0.0
100.0
true
true
"" ""
PENS
"High Income" 1.0 0 -16777216 true "" "ifelse count cars with [parked? = true and park <= parking-cars-percentage and income-group = 2] > 0 [plot (count cars with [parked? = true and income-group = 2] / count cars with [income-group = 2 and wants-to-park]) * 100][plot 0]"
"Middle Income" 1.0 0 -13791810 true "" "ifelse count cars with [parked? = true and park <= parking-cars-percentage and income-group = 1] > 0 [plot (count cars with [parked? = true and income-group = 1] / count cars with [income-group = 1 and wants-to-park]) * 100][plot 0]"
"Low Income" 1.0 0 -2674135 true "" "ifelse count cars with [parked? = true and park <= parking-cars-percentage and income-group = 0] > 0 [plot (count cars with [parked? = true and income-group = 0] / count cars with [income-group = 0 and wants-to-park]) * 100][ plot 0]"

SLIDER
15
770
270
803
fines-multiplier
fines-multiplier
1
20
4.0
1
1
time(s)
HORIZONTAL

TEXTBOX
15
725
281
761
How high should the fines be in terms of the original hourly fee?
13
0.0
1

MONITOR
1700
125
1800
170
Number of Cars
count cars
17
1
11

PLOT
1860
1235
2265
1535
Share of Income Class on Yellow Lot
Time
%
0.0
7200.0
0.0
100.0
true
true
"" ""
PENS
"High Income" 1.0 0 -16777216 true "" "ifelse count cars > 0 and count cars-on yellow-lot > 0 [plot (count cars with [([pcolor] of patch-here = [255.0 254.997195 102.02397]) and income-group = 2] / count cars-on yellow-lot) * 100] [plot 0]"
"Middle Income" 1.0 0 -13791810 true "" "ifelse count cars > 0 and count cars-on yellow-lot > 0 [plot (count cars with [([pcolor] of patch-here = [255.0 254.997195 102.02397]) and income-group = 1] / count cars-on yellow-lot) * 100] [plot 0]"
"Low Income" 1.0 0 -2674135 true "" "ifelse count cars > 0 and count cars-on yellow-lot > 0 [plot (count cars with [([pcolor] of patch-here = [255.0 254.997195 102.02397]) and income-group = 0] / count cars-on yellow-lot) * 100][plot 0]"

TEXTBOX
20
820
260
868
How often every hour should one of the lots be controlled?
13
0.0
1

SLIDER
20
865
270
898
controls-per-hour
controls-per-hour
1
8
1.0
1
1
time(s)
HORIZONTAL

PLOT
1448
323
1854
633
Dynamic Fee of Different Lots
Time
Euro
0.0
7200.0
0.0
5.0
true
true
"set-plot-background-color grey - 2" ""
PENS
"Yellow Lot" 1.0 0 -855445 true "" "plot yellow-lot-current-fee"
"Teal Lot" 1.0 0 -14520940 true "" "plot teal-lot-current-fee"
"Green Lot" 1.0 0 -8732573 true "" "plot green-lot-current-fee"
"Blue Lot" 1.0 0 -13740902 true "" "plot blue-lot-current-fee"

SWITCH
5
110
140
143
demo-mode
demo-mode
1
1
-1000

SLIDER
5
280
270
313
target-start-occupancy
target-start-occupancy
0
1
0.8120698015158538
0.05
1
NIL
HORIZONTAL

SLIDER
21
945
271
978
temporal-resolution
temporal-resolution
0
3600
1700.0
100
1
NIL
HORIZONTAL

TEXTBOX
80
910
230
938
How many ticks should be considered equal to one hour?
11
0.0
1

SLIDER
140
170
270
203
num-garages
num-garages
0
5
2.0
1
1
NIL
HORIZONTAL

SWITCH
5
370
270
403
dynamic-pricing-baseline
dynamic-pricing-baseline
0
1
-1000

SLIDER
5
310
270
343
parking-cars-percentage
parking-cars-percentage
0
100
77.73591064639997
1
1
%
HORIZONTAL

PLOT
2302
18
2733
316
Share of Vanished Cars per Income Class
Time
%
0.0
10.0
0.0
100.0
true
true
"" ""
PENS
"Low Income" 1.0 0 -2674135 true "" "if initial-count-low > 0 [ plot (vanished-cars-low / initial-count-low) * 100]"
"Middle Income" 1.0 0 -13345367 true "" "if initial-count-middle  > 0 [plot (vanished-cars-middle / initial-count-middle) * 100]"
"High Income" 1.0 0 -16777216 true "" "if initial-count-high  > 0 [plot (vanished-cars-high / initial-count-high) * 100]"

INPUTBOX
20
985
270
1045
output-turtle-file-path
test.csv
1
0
String

SWITCH
140
110
270
143
document-turtles
document-turtles
1
1
-1000

SLIDER
5
340
270
373
demand-curve-intercept
demand-curve-intercept
0
0.25
0.25
0.01
1
NIL
HORIZONTAL

PLOT
2301
647
2706
927
Demand
NIL
NIL
0.0
21600.0
0.0
100.0
true
false
"" ""
PENS
"default" 1.0 0 -16777216 true "" "plot parking-cars-percentage"

SWITCH
140
140
270
173
group-pricing
group-pricing
1
1
-1000

PLOT
1448
640
1688
790
Group Fees Yellow Lot
NIL
NIL
0.0
10.0
0.0
10.0
true
true
"set-plot-background-color [255.0 254.997195 102.02397]" ""
PENS
"Low Income" 1.0 0 -2674135 true "" "if group-pricing [plot item 0 [group-fees] of one-of yellow-lot]"
"Middle Income" 1.0 0 -13791810 true "" "if group-pricing [plot item 1 [group-fees] of one-of yellow-lot]"
"High Income" 1.0 0 -16777216 true "" "if group-pricing [plot item 2 [group-fees] of one-of yellow-lot]"

PLOT
1448
796
1688
941
Group Fees Green Lot
NIL
NIL
0.0
10.0
0.0
10.0
true
true
"set-plot-background-color [122.92632 173.61190499999998 116.145105]" ""
PENS
"Low Income" 1.0 0 -2674135 true "" "if group-pricing [plot item 0 [group-fees] of one-of green-lot]"
"Middle Income" 1.0 0 -13791810 true "" "if group-pricing [plot item 1 [group-fees] of one-of green-lot]"
"High Income" 1.0 0 -16777216 true "" "if group-pricing [plot item 2 [group-fees] of one-of green-lot]"

PLOT
1692
640
1917
790
Group Fees Teal Lot
NIL
NIL
0.0
10.0
0.0
10.0
true
true
"set-plot-background-color [57.189615 106.713675 147.774285]" ""
PENS
"Low Income" 1.0 0 -2674135 true "" "if group-pricing [plot item 0 [group-fees] of one-of teal-lot]"
"Middle Income" 1.0 0 -13791810 true "" "if group-pricing [plot item 1 [group-fees] of one-of teal-lot]"
"High income" 1.0 0 -16777216 true "" "if group-pricing [plot item 2 [group-fees] of one-of teal-lot]"

PLOT
1692
796
1917
941
Group Fees Blue Lot
NIL
NIL
0.0
10.0
0.0
10.0
true
true
"set-plot-background-color [25.867455 51.02805 178.54946999999999]" ""
PENS
"Low Income" 1.0 0 -2674135 true "" "if group-pricing [plot item 0 [group-fees] of one-of blue-lot]"
"Middle Income" 1.0 0 -13791810 true "" "if group-pricing [plot item 1 [group-fees] of one-of blue-lot]"
"High Income" 1.0 0 -16777216 true "" "if group-pricing [plot item 2 [group-fees] of one-of blue-lot]"

SLIDER
20
1045
270
1078
min-util
min-util
-10
5
-10.0
0.5
1
NIL
HORIZONTAL

PLOT
1449
952
1854
1232
Outcome per Income Class
NIL
NIL
0.0
10.0
0.0
0.0
true
true
"" ""
PENS
"High Income" 1.0 0 -16777216 true "" "if count cars with [income-group = 2 ] > 0 [plot mean [outcome] of cars with [income-group = 2 and outcome != -99]]"
"Middle Income" 1.0 0 -13791810 true "" "if count cars with [income-group = 1 ] > 0 [plot mean [outcome] of cars with [income-group = 1 and outcome != -99]]"
"Low Income" 1.0 0 -2674135 true "" "if count cars with [income-group = 0 ] > 0 [plot mean [outcome] of cars with [income-group = 0 and outcome != -99]]"
"Global" 1.0 0 -7500403 true "" "if count cars with [income-group = 0 ] > 0 [plot mean [outcome] of cars with [outcome != -99]]"

SWITCH
5
170
140
203
debugging
debugging
1
1
-1000

INPUTBOX
20
1085
237
1145
synthetic-population-file-path
synthetic_data.csv
1
0
String

SWITCH
5
205
270
238
use-synthetic-population
use-synthetic-population
0
1
-1000

@#$#@#$#@
# WHAT IS IT?

This is a model of traffic moving in a city grid. A portion of the agents tries to park on the curbside or the parking garages. The model is based on the traffic grid model by (Wilensky, 2003) and the seminar paper by (Aziz et al., 2020).



# Environment

The model’s environment is defined by a grid layout of roads and blocks. Located at the curbside, the yellow, green, teal, and blue patches designate parking spaces that are randomly scattered across the grid. Stripes of parking places situated opposite to one another are grouped. The coloring, indicating the different CPZs (Controlled Parking Zones) in the model, is then assigned depending on the distance of the groups to the center of the map, with the brightness of the colors decreasing the larger this distance grows. Due to their centrality, the green and, mainly, the yellow CPZ can be interpreted as most closely resembling the Central Business District (CBD) of the simulated city center. Beyond that, this model also introduces parking garages to account for off-street parking represented by the large blocks of black patches scattered across the map.

# Agents and Attributes

The central agents of our model are cars moving across the grid. In particular, 90% of all vehicles look for parking, and the remainder traverses the grid. For each car cruising for parking, a random parking duration is drawn from a gamma distribution, which was adopted from (Jakob & Menendez, 2021).

### Income

For every car, an income is randomly drawn from a log-normal distribution. Related work shows that the income of up to 99% of the population can be approximated with a two-parameter log-normal distribution (Clementi & Gallegati, 2005).


We calibrated the two parameters (the population mean and the population median) following the income distribution of the country our model city is located in. Based on the standard deviation, cars are divided into three income classes: Incomes within one standard deviation of the mean (68.2% of the population) are assigned the “middle-income” class. Deviations above or below this mark are called “high-” or “low-income”, (both 15.8% of the population) respectively.

### Willingness to Pay (WTP)

This variable captures the amount of money a driver is willing to pay for parking per
hour. Although income is a significant determinant of WTP, studies stress the importance of behavioral factors such as perceived comfort and security. To account for this individual variance, we randomly draw the WTP for each driver from a gamma distribution dependent on their income class.

Due to a lack of empirical evidence, the parameters of the distribution were manually calibrated to preserve the correlation between income and WTP and to ensure the functioning of the underlying parking routines in the model, i.e. to avoid excluding low-income drivers completely from parking: While for low-income drivers a mean of 2.5€ per hour was selected, the means for middle- and high-income cars amount to 4.5€ and 8€ per hour, respectively. Concerning the variance, for every income level, it was calibrated to correspond to the respective variance of the class-specific income distribution relative to its mean, mimicking the shapes of the individual intra-class income distributions. Specifying whether they are willing to park on a parking place without paying, the agents own the attribute parking-offender?.


# Behavioral Rules

### Navigation

All cars navigate the grid with previously assigned goals. For traversing cars, this goal is one of the exit points of the street network. For cars seeking to park, destinations are assigned with probabilities inversely proportional to their distance to the center of the grid, accounting for the higher popularity of the CBD. After target assignment, cars curate a list of the closest parking opportunities and elect the shortest route to the first one according to the NetLogo network extension.4 Upon arrival, cars attempt to park in the road of their assigned target. If no spot is available (or if spots are too expensive), cars move to the next list item. Similar to (Shoup, 2011), garages are only considered if there is no curbside parking at a cheaper cost since curbside parking is generally considered the more attractive option.

### Parking

Once a car has entered the street with its parking location of choice, it requests the fee of the closest available CPZ. If the fee is within the WTP, the car parks and the fee is added to the municipality revenues; if the fee exceeds the WTP, the driver will resume searching. We assume that WTP increases proportionally to the time spent cruising for parking. If a car belongs to the parking offenders, it will calculate the probability of getting caught. For this calculation, we assume rational actors with complete knowledge of the environment (i.e., the number of controls per hour and the fine as a multiple of the parking price are known a priori to offenders). Upon completion of their parking time, cars leave the CPZ and navigate towards the edge of the grid, where they are replaced with newly set up cars. In contrast, cars unable to find parking are not replaced once they leave the map. This preserves the change to the social distribution in the model that this behavior introduces.

# Sources

Aziz, M., Daube, J., Exner, P., Gutmann, J., Klenk, J., & Vyas, A. (2020). An Agent-Based Model for Simulating the Effect of Parking Prices [Seminar Report]. University of Mannheim.

Clementi, F., & Gallegati, M. (2005). Pareto’s Law of Income Distribution: Evidence for Germany, the United Kingdom, and the United States. In A. Chatterjee, S. Yarlagadda, & B. K. Chakrabarti (Eds.), Econophysics of Wealth Distributions (pp. 3–14). Springer Milan. https://doi.org/10.1007/88-470-0389-X_1


Jakob, M., & Menendez, M. (2021). Parking Pricing Vs. Congestion Pricing: A Macroscopic Analysis of Their Impact on Traffic. Transportmetrica A: Transport Science, 17(4), 462–491. https://doi.org/10.1080/23249935.2020.1797924

Shoup, D. (2011). The High Cost of Free Parking (Updated). Planners Press, American Planning Association.

Wilensky, U. (2003). Netlogo Traffic Grid Model. Center for Connected Learning and Computer-Based Modeling, Northwestern University. http://ccl.northwestern.edu/netlogo/models/TrafficGrid



<!-- 2022 -->
@#$#@#$#@
default
true
0
Polygon -7500403 true true 150 5 40 250 150 205 260 250

airplane
true
0
Polygon -7500403 true true 150 0 135 15 120 60 120 105 15 165 15 195 120 180 135 240 105 270 120 285 150 270 180 285 210 270 165 240 180 180 285 195 285 165 180 105 180 60 165 15

arrow
true
0
Polygon -7500403 true true 150 0 0 150 105 150 105 293 195 293 195 150 300 150

box
false
0
Polygon -7500403 true true 150 285 285 225 285 75 150 135
Polygon -7500403 true true 150 135 15 75 150 15 285 75
Polygon -7500403 true true 15 75 15 225 150 285 150 135
Line -16777216 false 150 285 150 135
Line -16777216 false 150 135 15 75
Line -16777216 false 150 135 285 75

bug
true
0
Circle -7500403 true true 96 182 108
Circle -7500403 true true 110 127 80
Circle -7500403 true true 110 75 80
Line -7500403 true 150 100 80 30
Line -7500403 true 150 100 220 30

butterfly
true
0
Polygon -7500403 true true 150 165 209 199 225 225 225 255 195 270 165 255 150 240
Polygon -7500403 true true 150 165 89 198 75 225 75 255 105 270 135 255 150 240
Polygon -7500403 true true 139 148 100 105 55 90 25 90 10 105 10 135 25 180 40 195 85 194 139 163
Polygon -7500403 true true 162 150 200 105 245 90 275 90 290 105 290 135 275 180 260 195 215 195 162 165
Polygon -16777216 true false 150 255 135 225 120 150 135 120 150 105 165 120 180 150 165 225
Circle -16777216 true false 135 90 30
Line -16777216 false 150 105 195 60
Line -16777216 false 150 105 105 60

car
true
0
Polygon -7500403 true true 180 15 164 21 144 39 135 60 132 74 106 87 84 97 63 115 50 141 50 165 60 225 150 285 165 285 225 285 225 15 180 15
Circle -16777216 true false 180 30 90
Circle -16777216 true false 180 180 90
Polygon -16777216 true false 80 138 78 168 135 166 135 91 105 106 96 111 89 120
Circle -7500403 true true 195 195 58
Circle -7500403 true true 195 47 58

car side
false
0
Polygon -7500403 true true 19 147 11 125 16 105 63 105 99 79 155 79 180 105 243 111 266 129 253 149
Circle -16777216 true false 43 123 42
Circle -16777216 true false 194 124 42
Polygon -16777216 true false 101 87 73 108 171 108 151 87
Line -8630108 false 121 82 120 108
Polygon -1 true false 242 121 248 128 266 129 247 115
Rectangle -16777216 true false 12 131 28 143

car top
true
0
Polygon -7500403 true true 151 8 119 10 98 25 86 48 82 225 90 270 105 289 150 294 195 291 210 270 219 225 214 47 201 24 181 11
Polygon -16777216 true false 210 195 195 210 195 135 210 105
Polygon -16777216 true false 105 255 120 270 180 270 195 255 195 225 105 225
Polygon -16777216 true false 90 195 105 210 105 135 90 105
Polygon -1 true false 205 29 180 30 181 11
Line -7500403 false 210 165 195 165
Line -7500403 false 90 165 105 165
Polygon -16777216 true false 121 135 180 134 204 97 182 89 153 85 120 89 98 97
Line -16777216 false 210 90 195 30
Line -16777216 false 90 90 105 30
Polygon -1 true false 95 29 120 30 119 11

circle
false
0
Circle -7500403 true true 0 0 300

circle 2
false
0
Circle -7500403 true true 0 0 300
Circle -16777216 true false 30 30 240

cow
false
0
Polygon -7500403 true true 200 193 197 249 179 249 177 196 166 187 140 189 93 191 78 179 72 211 49 209 48 181 37 149 25 120 25 89 45 72 103 84 179 75 198 76 252 64 272 81 293 103 285 121 255 121 242 118 224 167
Polygon -7500403 true true 73 210 86 251 62 249 48 208
Polygon -7500403 true true 25 114 16 195 9 204 23 213 25 200 39 123

cylinder
false
0
Circle -7500403 true true 0 0 300

dot
false
0
Circle -7500403 true true 90 90 120

face happy
false
0
Circle -7500403 true true 8 8 285
Circle -16777216 true false 60 75 60
Circle -16777216 true false 180 75 60
Polygon -16777216 true false 150 255 90 239 62 213 47 191 67 179 90 203 109 218 150 225 192 218 210 203 227 181 251 194 236 217 212 240

face neutral
false
0
Circle -7500403 true true 8 7 285
Circle -16777216 true false 60 75 60
Circle -16777216 true false 180 75 60
Rectangle -16777216 true false 60 195 240 225

face sad
false
0
Circle -7500403 true true 8 8 285
Circle -16777216 true false 60 75 60
Circle -16777216 true false 180 75 60
Polygon -16777216 true false 150 168 90 184 62 210 47 232 67 244 90 220 109 205 150 198 192 205 210 220 227 242 251 229 236 206 212 183

fish
false
0
Polygon -1 true false 44 131 21 87 15 86 0 120 15 150 0 180 13 214 20 212 45 166
Polygon -1 true false 135 195 119 235 95 218 76 210 46 204 60 165
Polygon -1 true false 75 45 83 77 71 103 86 114 166 78 135 60
Polygon -7500403 true true 30 136 151 77 226 81 280 119 292 146 292 160 287 170 270 195 195 210 151 212 30 166
Circle -16777216 true false 215 106 30

flag
false
0
Rectangle -7500403 true true 60 15 75 300
Polygon -7500403 true true 90 150 270 90 90 30
Line -7500403 true 75 135 90 135
Line -7500403 true 75 45 90 45

flower
false
0
Polygon -10899396 true false 135 120 165 165 180 210 180 240 150 300 165 300 195 240 195 195 165 135
Circle -7500403 true true 85 132 38
Circle -7500403 true true 130 147 38
Circle -7500403 true true 192 85 38
Circle -7500403 true true 85 40 38
Circle -7500403 true true 177 40 38
Circle -7500403 true true 177 132 38
Circle -7500403 true true 70 85 38
Circle -7500403 true true 130 25 38
Circle -7500403 true true 96 51 108
Circle -16777216 true false 113 68 74
Polygon -10899396 true false 189 233 219 188 249 173 279 188 234 218
Polygon -10899396 true false 180 255 150 210 105 210 75 240 135 240

house
false
0
Rectangle -7500403 true true 45 120 255 285
Rectangle -16777216 true false 120 210 180 285
Polygon -7500403 true true 15 120 150 15 285 120
Line -16777216 false 30 120 270 120

leaf
false
0
Polygon -7500403 true true 150 210 135 195 120 210 60 210 30 195 60 180 60 165 15 135 30 120 15 105 40 104 45 90 60 90 90 105 105 120 120 120 105 60 120 60 135 30 150 15 165 30 180 60 195 60 180 120 195 120 210 105 240 90 255 90 263 104 285 105 270 120 285 135 240 165 240 180 270 195 240 210 180 210 165 195
Polygon -7500403 true true 135 195 135 240 120 255 105 255 105 285 135 285 165 240 165 195

line
true
0
Line -7500403 true 150 0 150 300

line half
true
0
Line -7500403 true 150 0 150 150

pentagon
false
0
Polygon -7500403 true true 150 15 15 120 60 285 240 285 285 120

person
false
0
Circle -7500403 true true 110 5 80
Polygon -7500403 true true 105 90 120 195 90 285 105 300 135 300 150 225 165 300 195 300 210 285 180 195 195 90
Rectangle -7500403 true true 127 79 172 94
Polygon -7500403 true true 195 90 240 150 225 180 165 105
Polygon -7500403 true true 105 90 60 150 75 180 135 105

plant
false
0
Rectangle -7500403 true true 135 90 165 300
Polygon -7500403 true true 135 255 90 210 45 195 75 255 135 285
Polygon -7500403 true true 165 255 210 210 255 195 225 255 165 285
Polygon -7500403 true true 135 180 90 135 45 120 75 180 135 210
Polygon -7500403 true true 165 180 165 210 225 180 255 120 210 135
Polygon -7500403 true true 135 105 90 60 45 45 75 105 135 135
Polygon -7500403 true true 165 105 165 135 225 105 255 45 210 60
Polygon -7500403 true true 135 90 120 45 150 15 180 45 165 90

square
false
0
Rectangle -7500403 true true 30 30 270 270

square 2
false
0
Rectangle -7500403 true true 30 30 270 270
Rectangle -16777216 true false 60 60 240 240

star
false
0
Polygon -7500403 true true 151 1 185 108 298 108 207 175 242 282 151 216 59 282 94 175 3 108 116 108

target
false
0
Circle -7500403 true true 0 0 300
Circle -16777216 true false 30 30 240
Circle -7500403 true true 60 60 180
Circle -16777216 true false 90 90 120
Circle -7500403 true true 120 120 60

tree
false
0
Circle -7500403 true true 118 3 94
Rectangle -6459832 true false 120 195 180 300
Circle -7500403 true true 65 21 108
Circle -7500403 true true 116 41 127
Circle -7500403 true true 45 90 120
Circle -7500403 true true 104 74 152

triangle
false
0
Polygon -7500403 true true 150 30 15 255 285 255

triangle 2
false
0
Polygon -7500403 true true 150 30 15 255 285 255
Polygon -16777216 true false 151 99 225 223 75 224

truck
false
0
Rectangle -7500403 true true 4 45 195 187
Polygon -7500403 true true 296 193 296 150 259 134 244 104 208 104 207 194
Rectangle -1 true false 195 60 195 105
Polygon -16777216 true false 238 112 252 141 219 141 218 112
Circle -16777216 true false 234 174 42
Rectangle -7500403 true true 181 185 214 194
Circle -16777216 true false 144 174 42
Circle -16777216 true false 24 174 42
Circle -7500403 false true 24 174 42
Circle -7500403 false true 144 174 42
Circle -7500403 false true 234 174 42

turtle
true
0
Polygon -10899396 true false 215 204 240 233 246 254 228 266 215 252 193 210
Polygon -10899396 true false 195 90 225 75 245 75 260 89 269 108 261 124 240 105 225 105 210 105
Polygon -10899396 true false 105 90 75 75 55 75 40 89 31 108 39 124 60 105 75 105 90 105
Polygon -10899396 true false 132 85 134 64 107 51 108 17 150 2 192 18 192 52 169 65 172 87
Polygon -10899396 true false 85 204 60 233 54 254 72 266 85 252 107 210
Polygon -7500403 true true 119 75 179 75 209 101 224 135 220 225 175 261 128 261 81 224 74 135 88 99

wheel
false
0
Circle -7500403 true true 3 3 294
Circle -16777216 true false 30 30 240
Line -7500403 true 150 285 150 15
Line -7500403 true 15 150 285 150
Circle -7500403 true true 120 120 60
Line -7500403 true 216 40 79 269
Line -7500403 true 40 84 269 221
Line -7500403 true 40 216 269 79
Line -7500403 true 84 40 221 269

x
false
0
Polygon -7500403 true true 270 75 225 30 30 225 75 270
Polygon -7500403 true true 30 75 75 30 270 225 225 270
@#$#@#$#@
NetLogo 6.2.2
@#$#@#$#@
@#$#@#$#@
@#$#@#$#@
<experiments>
  <experiment name="experiment" repetitions="1" runMetricsEveryStep="true">
    <setup>setup</setup>
    <go>go</go>
    <timeLimit steps="21600"/>
    <metric>count turtles</metric>
    <enumeratedValueSet variable="hide-nodes">
      <value value="true"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="demo-mode">
      <value value="false"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="target-start-occupancy">
      <value value="0.5"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="blue-lot-fee">
      <value value="2"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="green-lot-fee">
      <value value="2"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="show-goals">
      <value value="false"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="pop-median-income">
      <value value="22713"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="yellow-lot-fee">
      <value value="2"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="controls-per-hour">
      <value value="2"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="fines-multiplier">
      <value value="5"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="ticks-per-cycle">
      <value value="20"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="num-garages">
      <value value="0"/>
      <value value="1"/>
      <value value="2"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="temporal-resolution">
      <value value="1800"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="lot-distribution-percentage">
      <value value="0.8"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="wtp-income-share">
      <value value="0.005"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="pop-mean-income">
      <value value="25882"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="num-cars">
      <value value="490"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="dynamic-pricing-baseline">
      <value value="true"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="orange-lot-fee">
      <value value="2"/>
    </enumeratedValueSet>
  </experiment>
</experiments>
@#$#@#$#@
@#$#@#$#@
default
0.0
-0.2 0 0.0 1.0
0.0 1 1.0 0.0
0.2 0 0.0 1.0
link direction
true
0
Line -7500403 true 150 150 90 180
Line -7500403 true 150 150 210 180
@#$#@#$#@
1
@#$#@#$#@

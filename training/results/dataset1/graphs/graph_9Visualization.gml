graph [
  directed 1
  levelsToS 3
  node [
    id 0
    label "0"
    schedulingPolicy "FIFO"
    bufferSizes "32000"
    schedulingWeights "-"
    levelsQoS 1
    tosToQoSqueue "0,1,2"
  ]
  node [
    id 1
    label "1"
    schedulingPolicy "FIFO"
    bufferSizes "32000"
    schedulingWeights "-"
    levelsQoS 1
    tosToQoSqueue "0,1,2"
  ]
  node [
    id 2
    label "2"
    schedulingPolicy "FIFO"
    bufferSizes "32000"
    schedulingWeights "-"
    levelsQoS 1
    tosToQoSqueue "0,1,2"
  ]
  node [
    id 3
    label "3"
    schedulingPolicy "FIFO"
    bufferSizes "32000"
    schedulingWeights "-"
    levelsQoS 1
    tosToQoSqueue "0,1,2"
  ]
  node [
    id 4
    label "4"
    schedulingPolicy "FIFO"
    bufferSizes "32000"
    schedulingWeights "-"
    levelsQoS 1
    tosToQoSqueue "0,1,2"
  ]
  node [
    id 5
    label "5"
    schedulingPolicy "FIFO"
    bufferSizes "32000"
    schedulingWeights "-"
    levelsQoS 1
    tosToQoSqueue "0,1,2"
  ]
  node [
    id 6
    label "6"
    schedulingPolicy "FIFO"
    bufferSizes "32000"
    schedulingWeights "-"
    levelsQoS 1
    tosToQoSqueue "0,1,2"
  ]
  node [
    id 7
    label "7"
    schedulingPolicy "FIFO"
    bufferSizes "32000"
    schedulingWeights "-"
    levelsQoS 1
    tosToQoSqueue "0,1,2"
  ]
  node [
    id 8
    label "8"
    schedulingPolicy "FIFO"
    bufferSizes "32000"
    schedulingWeights "-"
    levelsQoS 1
    tosToQoSqueue "0,1,2"
  ]
  edge [
    source 0
    target 5
    bandwidth 100000
    port 0
  ]
  edge [
    source 0
    target 4
    bandwidth 100000
    port 1
  ]
  edge [
    source 0
    target 8
    bandwidth 100000
    port 2
  ]
  edge [
    source 0
    target 6
    bandwidth 100000
    port 3
  ]
  edge [
    source 1
    target 3
    bandwidth 100000
    port 0
  ]
  edge [
    source 1
    target 4
    bandwidth 100000
    port 1
  ]
  edge [
    source 2
    target 3
    bandwidth 100000
    port 0
  ]
  edge [
    source 2
    target 8
    bandwidth 100000
    port 1
  ]
  edge [
    source 2
    target 5
    bandwidth 100000
    port 2
  ]
  edge [
    source 2
    target 6
    bandwidth 100000
    port 3
  ]
  edge [
    source 3
    target 1
    bandwidth 100000
    port 0
  ]
  edge [
    source 3
    target 2
    bandwidth 100000
    port 1
  ]
  edge [
    source 4
    target 0
    bandwidth 100000
    port 0
  ]
  edge [
    source 4
    target 1
    bandwidth 100000
    port 1
  ]
  edge [
    source 4
    target 6
    bandwidth 100000
    port 2
  ]
  edge [
    source 4
    target 8
    bandwidth 100000
    port 3
  ]
  edge [
    source 5
    target 0
    bandwidth 100000
    port 0
  ]
  edge [
    source 5
    target 2
    bandwidth 100000
    port 1
  ]
  edge [
    source 5
    target 7
    bandwidth 100000
    port 2
  ]
  edge [
    source 6
    target 0
    bandwidth 100000
    port 0
  ]
  edge [
    source 6
    target 2
    bandwidth 100000
    port 1
  ]
  edge [
    source 6
    target 4
    bandwidth 100000
    port 2
  ]
  edge [
    source 6
    target 8
    bandwidth 100000
    port 3
  ]
  edge [
    source 7
    target 5
    bandwidth 100000
    port 0
  ]
  edge [
    source 7
    target 8
    bandwidth 100000
    port 1
  ]
  edge [
    source 8
    target 0
    bandwidth 100000
    port 0
  ]
  edge [
    source 8
    target 2
    bandwidth 100000
    port 1
  ]
  edge [
    source 8
    target 4
    bandwidth 100000
    port 2
  ]
  edge [
    source 8
    target 6
    bandwidth 100000
    port 3
  ]
  edge [
    source 8
    target 7
    bandwidth 100000
    port 4
  ]
]

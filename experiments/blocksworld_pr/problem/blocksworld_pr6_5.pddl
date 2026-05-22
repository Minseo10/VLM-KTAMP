(define (problem blocksworld_pr6_5)
  (:domain blocksworld-original)
  (:objects
    cyan red grey yellow magenta brown
  )
  (:init
    (arm-empty)
    (on-table cyan)
    (on red cyan)
    (clear red)
    (on-table grey)
    (on yellow grey)
    (clear yellow)
    (on-table magenta)
    (on brown magenta)
    (clear brown)
  )
  (:goal
    (and
      (on-table brown)
      (on magenta brown)
      (on yellow magenta)
      (on-table grey)
      (on cyan grey)
      (on red cyan)
    )
  )
)
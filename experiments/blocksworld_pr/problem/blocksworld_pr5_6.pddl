(define (problem blocksworld_pr5_3)
  (:domain blocksworld-original)
  (:objects
    green red yellow blue white
  )
  (:init
    (arm-empty)
    (on-table green)
    (on red green)
    (on yellow red)
    (clear yellow)
    (on-table blue)
    (on white blue)
    (clear white)
  )
  (:goal
    (and
      (on-table yellow)
      (on green yellow)
      (on-table red)
      (on blue red)
      (on white blue)
    )
  )
)
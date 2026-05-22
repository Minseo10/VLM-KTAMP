(define (problem blocksworld_pr5_1)
  (:domain blocksworld-original)
  (:objects
    white red yellow blue magenta
  )
  (:init
    (arm-empty)
    (on-table white)
    (on red white)
    (on yellow red)
    (clear yellow)
    (on-table blue)
    (on magenta blue)
    (clear magenta)
  )
  (:goal
    (and
      (on-table red)
      (on yellow red)
      (on magenta yellow)
      (on-table white)
      (on blue white)
    )
  )
)
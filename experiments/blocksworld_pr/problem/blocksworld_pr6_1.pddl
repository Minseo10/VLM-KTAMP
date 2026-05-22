(define (problem blocksworld_pr6_1)
  (:domain blocksworld-original)
  (:objects
    grey cyan red yellow blue brown
  )
  (:init
    (arm-empty)
    (on-table grey)
    (on cyan grey)
    (on red cyan)
    (clear red)
    (on-table yellow)
    (on blue yellow)
    (on brown blue)
    (clear brown)
  )
  (:goal
    (and
      (on-table red)
      (on yellow red)
      (on blue yellow)
      (on cyan blue)
      (on-table brown)
      (on grey brown)
    )
  )
)
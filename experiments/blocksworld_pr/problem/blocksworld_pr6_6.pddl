(define (problem blocksworld_pr6_3)
  (:domain blocksworld-original)
  (:objects
    green apricot red yellow blue white
  )
  (:init
    (arm-empty)
    (on-table green)
    (on apricot green)
    (on red apricot)
    (clear red)
    (on-table yellow)
    (on blue yellow)
    (on white blue)
    (clear white)
  )
  (:goal
    (and
      (on-table apricot)
      (on yellow apricot)
      (on blue yellow)
      (on-table white)
      (on red white)
      (on green red)
    )
  )
)
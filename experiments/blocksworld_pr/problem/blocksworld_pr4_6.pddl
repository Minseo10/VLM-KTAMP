(define (problem blocksworld_pr4_4)
  (:domain blocksworld-original)
  (:objects
    blue apricot yellow green
  )
  (:init
    (arm-empty)
    (on-table blue)
    (on apricot blue)
    (on yellow apricot)
    (on green yellow)
    (clear green)
  )
  (:goal
    (and
      (on-table green)
      (on yellow green)
      (on apricot yellow)
      (on blue apricot)
    )
  )
)
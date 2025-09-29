import scripts.application

if __name__ == "__main__":
    from scripts.scenes import dragon_scene_animated
    scripts.application.set_scene(dragon_scene_animated) 

    scripts.application.HEADLESS = False

    app = scripts.application.Application()
    app.start()

  
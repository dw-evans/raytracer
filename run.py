import scripts.application

if __name__ == "__main__":
    from scripts.scenes import dragon_scene
    scripts.application.set_scene(dragon_scene) 

    scripts.application.HEADLESS = True

    app = scripts.application.Application()
    app.start()

  
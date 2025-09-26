import scripts.application


if __name__ == "__main__":
    scripts.application.HEADLESS = True
    app = scripts.application.Application()
    app.start()

  
  
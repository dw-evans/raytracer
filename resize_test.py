import pygame
import moderngl
from pygame.locals import DOUBLEBUF, OPENGL, RESIZABLE

def main():
    pygame.init()
    pygame.display.set_caption("Resizable ModernGL Window")

    # Create resizable window with OpenGL context
    screen = pygame.display.set_mode(
        (800, 600),
        DOUBLEBUF | OPENGL | RESIZABLE
    )

    # Create ModernGL context from pygame's OpenGL context
    ctx = moderngl.create_context()

    # Set initial viewport
    width, height = screen.get_size()
    ctx.viewport = (0, 0, width, height)

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            elif event.type == pygame.VIDEORESIZE:
                width, height = event.size
                # Resize window
                screen = pygame.display.set_mode(
                    (width, height),
                    DOUBLEBUF | OPENGL | RESIZABLE
                )
                # Update ModernGL viewport
                ctx.viewport = (0, 0, width, height)

        # Clear screen with a color
        ctx.clear(0.1, 0.2, 0.3, 1.0)

        # (draw stuff here using ModernGL)

        pygame.display.flip()

    pygame.quit()

if __name__ == "__main__":
    main()

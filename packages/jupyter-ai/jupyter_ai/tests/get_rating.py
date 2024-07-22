import curses
import os
import textwrap
from typing import Any, Generator

from jupyter_ai.tests.utils import Gen, StreamingData # type: ignore

KEYS_ENTER = {curses.KEY_ENTER, ord('\n'), ord('\r'), ord(' ')}

def get_line_length() -> int:
    """
    Get the number of columns in the terminal.

    Returns:
        int: The number of columns in the terminal.
    """
    return os.get_terminal_size().columns

DEFAULT_LINE_LENGTH = get_line_length()

def get_title(title: str, line_length: int = DEFAULT_LINE_LENGTH) -> Generator[str, Any, None]:
    """
    Generate a title formatted with a border for display in the terminal.

    Args:
        title (str): The title text to display.
        line_length (int, optional): The length of the line for formatting. Defaults to DEFAULT_LINE_LENGTH.

    Yields:
        Generator[str, Any, None]: Formatted lines of the title.
    """
    usable_length = line_length - 6

    yield '  +' + '-' * usable_length + '+  '
    yield '  |{: ^{usable_length}}|  '.format(title, usable_length=usable_length)
    yield '  +' + '-' * usable_length + '+  '

def get_string(string: str, start_at: int = 0, num_lines: int = -1, line_length: int = DEFAULT_LINE_LENGTH, right_padding: int = 0) -> Generator[str, Any, int]:
    """
    Generate formatted lines of a string for display in the terminal.

    Args:
        string (str): The string to format.
        start_at (int, optional): The line number to start from. Defaults to 0.
        num_lines (int, optional): The number of lines to generate. Defaults to -1 (all lines).
        line_length (int, optional): The length of the line for formatting. Defaults to DEFAULT_LINE_LENGTH.
        right_padding (int, optional): The number of spaces to pad on the right side of the line. Defaults to 0.

    Yields:
        Generator[str, Any, int]: Formatted lines of the string.
    """
    usable_length = line_length - 6
    padded_length = usable_length - 4

    yield '  |' + ' ' * usable_length + '|  '

    line_on = 0
    num_printed = 0

    for line in string.splitlines():
        if not line: # handles blank lines
            line_on += 1

            if num_lines != -1 and (line_on <= start_at or line_on > start_at + num_lines):
                continue

            yield '  |' + ' ' * usable_length + '|  '
            num_printed += 1
            continue

        # uses python's built in text wrapping to format the line
        for line_segment in textwrap.wrap(line, padded_length - right_padding, replace_whitespace=False, drop_whitespace=False, tabsize=4):
            line_on += 1

            if num_lines != -1 and (line_on <= start_at or line_on > start_at + num_lines):
                continue

            yield '  |  ' + line_segment + ' ' * (padded_length - len(line_segment)) + '  |  '
            num_printed += 1
            
    yield '  |' + ' ' * usable_length + '|  '

    # if start_at is too large, provides extra padding so that the size of the text box does not change
    num_to_print = min(num_lines, line_on)

    if num_lines != -1:
        while num_printed < num_to_print:
            yield '  |' + ' ' * usable_length + '|  '
            num_printed += 1

    yield '  +' + '-' * usable_length + '+  '

    return line_on

class GetRating:
    """
    Class for handling user rating input in a terminal UI using the curses library.

    It provides a cross-platform interactive terminal interface for efficiently (and anonymously) comparing the LLM response to the expected response and rating it.

    Usage (as shown in the UI):
        Use the left and right arrow keys to select a rating, and enter to submit.
        Use the up and down arrow keys to scroll through the AI response if it is too long.
        Use the w and s keys to scroll through the prompt itself if it is too long.        

    The advantages over editing the CSV directly after the fact are:
        1. The user can start rating as soon as any LLM response is available, without needing to wait for them all to complete.
        2. It is very hard to view responses in a CSV file without proper formatting.
        3. This UI allows the models to be compared anonymously.
        4. It is less likely for the user to accidentally rate the wrong response given this UI.
        5. The UI automatically and instantly populates the next LLM response into the same format, which would not happen manually.
    """

    prompt_line_on: int = 0
    """The first line of the prompt to be displayed, equivalently how many lines of the prompt to skip. Used for scrolling."""

    resp_line_on: int = 0
    """The first line of the response to be displayed, equivalently how many lines of the response to skip. Used for scrolling."""

    max_prompt_line_on: int = 0
    """The maximum number of lines that the prompt can be scrolled to. Depends on terminal size."""

    max_resp_line_on: int = 0
    """The maximum number of lines that the response can be scrolled to. Depends on terminal size."""

    current_rating: int = 0
    options: list[str] = [
        'Unacceptable',
        'Low Quality',
        'Helpful',
        'Human Level',
        'Beyond',
    ]

    def __init__(self, data: StreamingData, expected: str, index: int, total: int):
        """
        Initialize the GetRating instance.

        Args:
            data (StreamingData): The data object containing the user input and AI response.
            expected (str): The expected response string.
            index (int): The current index of the rating in the sequence.
            total (int): The total number of ratings.
        """
        self.data = data
        self.index = index
        self.total = total
        self.expected = expected

    def _draw_option(self, screen: 'curses._CursesWindow', index: int, x: int, y: int): # type: ignore
        """
        Draw a rating option on the screen. If the option is selected, it will be highlighted.

        Args:
            screen (curses._CursesWindow): The curses screen object.
            index (int): The index of the rating option to draw.
            x (int): The x-coordinate to start drawing.
            y (int): The y-coordinate to start drawing.
        """
        option = self.options[index]
        text_attr = curses.A_STANDOUT if self.current_rating == index else curses.A_NORMAL
        border_attr = curses.A_STANDOUT if self.current_rating == index else curses.A_NORMAL

        screen.addstr(y, x, '+' + '-' * len(option) + '+', border_attr)
        y += 1
        screen.addstr(y, x, '|', border_attr)
        screen.addstr(y, x + 1, option, text_attr)
        screen.addstr(y, x + len(option) + 1, '|', border_attr)
        y += 1
        screen.addstr(y, x, '+' + '-' * len(option) + '+', border_attr)

    def draw_not_enough_lines(self, screen: 'curses._CursesWindow', num_missing: int=-1): # type: ignore
        """
        Display a message indicating that there are not enough lines in the terminal to draw the UI.

        Args:
            screen (curses._CursesWindow): The curses screen object.
            num_missing (int, optional): The number of missing lines. Defaults to -1.
        """
        screen.clear()

        screen.addstr(0, 0, 'Not enough rows in terminal to draw UI')
        screen.addstr(1, 0, 'Please expand your terminal and/or decrease its font size')
        if num_missing > 0:
            screen.addstr(2, 0, f'Missing {num_missing} row/s')

    def draw(self, screen: 'curses._CursesWindow'): # type: ignore
        """
        Draw the entire UI, including the user input, expected response, and AI response sections.

        Args:
            screen (curses._CursesWindow): The curses screen object.
        """
        screen.clear()

        max_y, max_x = screen.getmaxyx()

        x, y = 0, 0
    
        try: # tries drawing the UI. If it fails to draw the UI due to not enough rows in terminal, it will call draw_not_enough_lines instead.
            # Starts by drawing the user input
            for line in get_title('User Input', max_x):
                screen.addstr(y, x, line)
                y += 1

            progress = f'{self.index}/{self.total}'
            screen.addstr(y - 2, 5, progress, curses.A_BOLD)

            stars = '★' * self.current_rating + '☆' * (len(self.options) - self.current_rating - 1)
            screen.addstr(y - 2, max_x - len(self.options) - 4, stars, curses.A_BOLD)
            
            max_num_lines = max(5, max_y // 5)
            gen = Gen(get_string(self.data.get_prompt(), self.prompt_line_on, max_num_lines, max_x, right_padding=2))
            start_y = y
            for line in gen:
                screen.addstr(y, x, line)
                y += 1
            
            end_y = y - 2
            self.max_prompt_line_on = gen.value - max_num_lines
            y += 1

            # Draws the scroll bar showing current scroll progress
            if gen.value > max_num_lines:
                screen.addstr(start_y, max_x - 5, 'W')
                screen.addstr(end_y, max_x - 5, 'S')

                progress = round(self.prompt_line_on / self.max_prompt_line_on * (end_y - start_y - 2))
                screen.addstr(start_y + progress + 1, max_x  - 5, '▒')

            # Next draws the expected response
            for line in get_title('Expected Response', max_x):
                screen.addstr(y, x, line)
                y += 1

            for line in get_string(self.expected, 0, -1, max_x):
                screen.addstr(y, x, line)
                y += 1
            
            y += 1

            # Finally draws the response provided by the LLM
            for line in get_title('AI Assistant Response', max_x):
                screen.addstr(y, x, line)
                y += 1
            
            max_num_lines = max_y - y - 7

            if max_num_lines < 1:
                self.draw_not_enough_lines(screen, 1 - max_num_lines)
                return

            gen = Gen(get_string(self.data.get_generation(), self.resp_line_on, max_num_lines, max_x, right_padding=2))
            start_y = y
            for line in gen:
                screen.addstr(y, x, line)
                y += 1

            end_y = y - 2
            self.max_resp_line_on = gen.value - max_num_lines
            y += 1

            # Draws the scroll bar showing the current scroll progress
            if gen.value > max_num_lines:
                screen.addstr(start_y, max_x - 5, '↑')
                screen.addstr(end_y, max_x - 5, '↓')

                progress = round(self.resp_line_on / self.max_resp_line_on * (end_y - start_y - 2))
                screen.addstr(start_y + progress + 1, max_x  - 5, '▒')

            padded_length = max_x - 4
            center_distance = padded_length / len(self.options)

            for i, option in enumerate(self.options):
                self._draw_option(screen, i, int(3 + i * center_distance + len(option) / 2), y)

            y += 3
        except Exception as e:
            if y >= max_y - 1:
                self.draw_not_enough_lines(screen)
            else:
                raise e

    def config_curses(self):
        """
        Configure curses settings such as using default colors and hiding the cursor.
        """
        try:
            # use the default colors of the terminal
            curses.use_default_colors()
            # hide the cursor
            curses.curs_set(0)
        except:
            # Curses failed to initialize color support, eg. when TERM=vt100
            curses.initscr()

    def run_loop(self, screen: 'curses._CursesWindow') -> int: # type: ignore
        """
        Run the main loop for handling user input and updating the UI.

        Args:
            screen (curses._CursesWindow): The curses screen object.

        Returns:
            int: The final rating chosen by the user.
        """
        while True:
            self.draw(screen)

            key = screen.getch()

            if key in KEYS_ENTER:
                return self.current_rating
            
            if key in {ord('w'), ord('W')}:
                self.prompt_line_on = max(0, self.prompt_line_on  - 1)
            
            if key in {ord('s'), ord('S')}:
                self.prompt_line_on = min(self.max_prompt_line_on, self.prompt_line_on + 1)

            if key in {ord('a'), curses.KEY_LEFT}:
                self.current_rating = max(0, self.current_rating - 1)
            
            if key in {ord('d'), curses.KEY_RIGHT}:
                self.current_rating = min(len(self.options) - 1, self.current_rating + 1)

            match key:
                case curses.KEY_UP:
                    self.resp_line_on = max(0, self.resp_line_on - 1)
                case curses.KEY_DOWN:
                    self.resp_line_on = min(self.max_resp_line_on, self.resp_line_on + 1)
                case _:
                    pass


    def _start(self, screen: 'curses._CursesWindow') -> int: # type: ignore
        """
        Start the curses application.

        Args:
            screen (curses._CursesWindow): The curses screen object.

        Returns:
            int: The final rating chosen by the user.
        """
        self.config_curses()
        return self.run_loop(screen)
    
    def start(self) -> int:
        """
        Wrapper to start the curses application.

        Returns:
            int: The final rating chosen by the user.
        """
        return curses.wrapper(self._start)

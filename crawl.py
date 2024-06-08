"""
entry point of the crawler
"""

from typing import List, Dict
import json

import requests
from bs4 import BeautifulSoup
from bs4.element import Tag


def find_chapters(soup: BeautifulSoup) -> List[List[Tag]]:
    """_summary_
    Args: soup (BeautifulSoup): _description_

    Returns: List[List[Tag]]: _description_

    """
    _chapters = soup.find_all("h2", class_="chapter")
    result: List[List[Tag]] = []
    for _chapter in _chapters:
        children: List[Tag] = _chapter.find_next_siblings()
        _result: List[Tag] = [_chapter]
        for child in children:
            _class = child.get("class")
            if isinstance(_class, list) and "chapter" in _class:
                break
            _result.append(child)
        result.append(_result)
    return result


def find_sections(_chapter: List[Tag]) -> List[List[Tag]]:
    """_summary_

    Args:
    chapters (List[List[Tag]]): _description_

    Returns:
    List[List[Tag]]: _description_
    """
    _sections: List[List[Tag]] = []
    for child in _chapter:
        _class = child.get("class")
        if isinstance(_class, list) and "section" in _class:
            _sections.append([child])
        else:
            try:
                _sections[-1].append(child)
            except IndexError:
                _sections.append([child])
    return _sections


if __name__ == "__main__":
    response = requests.get("https://ffmpeg.org/ffmpeg.html", timeout=10)
    soup = BeautifulSoup(response.content, "html.parser")
    chapters = find_chapters(soup=soup)

    output: List[Dict[str, str]] = []

    for chapter in chapters:
        # find all sections of the chapter
        sections = find_sections(chapter)
        _chapter_dict: Dict[str, str] = {}
        if len(sections) > 1:
            for section in sections:
                # is the section name
                _chapter_dict[section[0].text] = "\n".join(
                    tag.text for tag in section
                )  # noqa
            output.append(_chapter_dict)
        elif len(sections) == 1:
            _chapter_dict["section"] = "\n".join(
                tag.text for tag in sections[0]
            )  # noqa
            output.append(_chapter_dict)

    with open("output.json", "w+", encoding="utf-8") as f:
        json.dump(output, f)
    print("done")

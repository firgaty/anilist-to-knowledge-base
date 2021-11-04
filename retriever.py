from __future__ import annotations
import requests
from typing import TypedDict, Dict, List, Optional, Set, Any, Iterator
from dataclasses import dataclass
from datetime import date, datetime
from collections import namedtuple
import json
import yaml
from pytablewriter import MarkdownTableWriter
from pathlib import Path
import argparse


@dataclass
class UserStatus:
    status: str
    progress: int
    score: int
    repeat: int
    notes: Optional[str]
    started_at: Optional[date]
    completed_at: Optional[date]
    updates_at: Optional[date]

    @classmethod
    def from_graphql(cls, raw: Dict[str, Any]) -> UserStatus:
        return UserStatus(
            status=raw["status"],
            progress=raw["progress"],
            score=raw["score"],
            repeat=raw["repeat"],
            notes=raw["notes"],
            started_at=parse_date(**raw["startedAt"]),
            completed_at=parse_date(**raw["completedAt"]),
            updates_at=datetime.fromtimestamp(raw["updatedAt"]).date(),
        )

    def to_md(self) -> str:
        table = [
            ["__Status__", f"{self.status}"],
            ["__Progress__", f"{self.progress}"],
            ["__Score__", f"{self.score}/10"],
            ["__Repeat__", f"{self.repeat}"],
            ["__Started at__", f"{self.started_at}"],
            ["__Completed at__", f"{self.completed_at}"],
            ["__Updated at__", f"{self.updates_at}"],
        ]

        writer = MarkdownTableWriter(
            headers=["Info", "Value"],
            value_matrix=table,
            margin=1,
        )

        return writer.dumps()


@dataclass
class AnimeRelation:
    anilist_id: int
    title: str
    type: str

    @classmethod
    def from_graphql(cls, raw: Dict[str, Any]) -> AnimeRelation:
        return AnimeRelation(
            raw["node"]["id"],
            raw["node"]["title"]["userPreferred"],
            raw["relationType"],
        )


@dataclass
class StudioRelation:
    studio_id: int
    name: str
    is_main: bool

    @classmethod
    def from_graphql(cls, raw: Dict[str, Any]) -> StudioRelation:
        return StudioRelation(
            studio_id=raw["node"]["id"],
            name=raw["node"]["name"],
            is_main=raw["isMain"],
        )


@dataclass
class Staff:
    staff_id: int
    name: str
    image: str

    @classmethod
    def from_graphql(cls, raw: Dict[str, Any]) -> Staff:
        return Staff(
            staff_id=raw["id"],
            name=raw["name"]["userPreferred"],
            image=raw["image"]["medium"],
        )


@dataclass
class Character:
    character_id: int
    name: str
    image: str

    @classmethod
    def from_graphql(cls, raw: Dict[str, Any]) -> Character:
        return Character(
            raw["id"],
            raw["name"]["userPreferred"],
            raw["image"]["medium"],
        )


@dataclass
class CharacterRelation:
    character: Character
    role: str
    voice_actors: List[Staff]

    @classmethod
    def from_graphql(slf, raw: Dict[str, Any]) -> CharacterRelation:
        return CharacterRelation(
            Character.from_graphql(raw["node"]),
            raw["role"],
            [Staff.from_graphql(x) for x in raw["voiceActors"]],
        )


@dataclass
class StaffRelation:
    role: str
    staff: Staff

    @classmethod
    def from_graphql(cls, raw: Dict[str, Any]) -> StaffRelation:
        return StaffRelation(
            role=raw["role"],
            staff=Staff.from_graphql(raw["node"]),
        )


@dataclass
class AnimeInfo:
    anilist_id: int
    mal_id: int
    is_adult: bool
    format: str
    is_favourite: bool
    title_romaji: Optional[str]
    title_english: Optional[str]
    title_native: Optional[str]
    title: str
    synonyms: List[str]
    duration: int
    season: str
    genres: List[str]
    tags: List[str]
    status: str
    season_year: int
    episodes: int
    description: str
    cover_image_extra_large: str
    cover_image_large: str
    cover_image_medium: str
    start_date: Optional[date]
    end_date: Optional[date]

    @classmethod
    def from_graphql(cls, raw: Dict[str, Any]) -> AnimeInfo:
        return AnimeInfo(
            raw["id"],
            raw["idMal"],
            raw["isAdult"],
            raw["format"],
            raw["isFavourite"],
            raw["title"]["romaji"],
            raw["title"]["english"],
            raw["title"]["native"],
            raw["title"]["userPreferred"],
            raw["synonyms"],
            raw["duration"],
            raw["season"],
            raw["genres"],
            [x["name"] for x in raw["tags"]],
            raw["status"],
            raw["seasonYear"],
            raw["episodes"],
            raw["description"],
            raw["coverImage"]["extraLarge"],
            raw["coverImage"]["large"],
            raw["coverImage"]["medium"],
            parse_date(**raw["startDate"]),
            parse_date(**raw["endDate"]),
        )

    def to_md(self) -> str:
        table = [
            ["__Anilist__", f"[link](https://anilist.co/anime/{self.anilist_id})"],
            ["__MyAnimeList__", f"[link](https://myanimelist.net/anime/{self.mal_id})"],
            ["__Cover__", f"![cover]({self.cover_image_medium})"],
            ["__English Title__", f"{self.title_english}"],
            ["__Romaji Title__", f"{self.title_romaji}"],
            ["__Native Title__", f"{self.title_native}"],
            ["__Synonyms__", f"{' '.join(self.synonyms)}"],
            ["__Year__", f"{self.season_year}"],
            ["__Season__", f"{self.season}"],
            ["__Status__", f"{self.status}"],
            ["__Episodes__", f"{self.episodes}"],
            ["__Duration__", f"{self.duration}"],
        ]

        writer = MarkdownTableWriter(
            headers=["Info", "Value"],
            value_matrix=table,
            margin=1,
        )

        return writer.dumps()


@dataclass
class Anime:
    info: AnimeInfo
    studios: List[StudioRelation]
    relations: List[AnimeRelation]
    characters: List[CharacterRelation]
    staffs: List[StaffRelation]
    user_status: UserStatus

    @classmethod
    def from_graphql(cls, raw: Dict[str, Any]) -> Anime:
        return Anime(
            AnimeInfo.from_graphql(raw["media"]),
            [StudioRelation.from_graphql(x) for x in raw["media"]["studios"]["edges"]],
            [AnimeRelation.from_graphql(x) for x in raw["media"]["relations"]["edges"]],
            [
                CharacterRelation.from_graphql(x)
                for x in raw["media"]["characters"]["edges"]
            ],
            [StaffRelation.from_graphql(x) for x in raw["media"]["staff"]["edges"]],
            UserStatus.from_graphql(raw),
        )

    def to_yaml(self) -> str:
        yaml_dict: Dict[str, Any] = {}

        yaml_dict["anime_info"] = self.info.__dict__
        yaml_dict["anime_status"] = self.user_status.__dict__
        yaml_dict["tags"] = ["Media/Anime", f"Media/Anime/{self.info.format}"]
        yaml_dict["aliases"] = [
            x
            for x in [
                self.info.title_english,
                self.info.title_native,
                self.info.title_romaji,
            ]
            if x is not None
        ]

        return yaml.dump(yaml_dict)

    def studios_to_md(self) -> str:
        studios = [
            [
                f"{x.name}",
                f"[link](https://anilist.co/studio/{x.studio_id})",
                "Y" if x.is_main else "N",
            ]
            for x in self.studios
        ]

        writer = MarkdownTableWriter(
            headers=["Name", "Link", "Main"],
            value_matrix=studios,
            margin=1,
        )

        return writer.dumps()

    def relations_to_md(self) -> str:
        relations = [
            [
                f"[[{sanitize_path(x.title)} ({x.anilist_id})|{x.title}]]",
                f"{x.type}",
                f"[link](https://anilist.co/anime/{x.anilist_id})",
            ]
            for x in self.relations
        ]

        writer = MarkdownTableWriter(
            headers=["Title", "Relation Type", "Link"],
            value_matrix=relations,
            margin=1,
        )

        return writer.dumps()

    def staff_to_md(self) -> str:
        relations = [
            [
                f"![image]({x.staff.image})",
                f"[[{sanitize_path(x.staff.name)} ({x.staff.staff_id})|{x.staff.name}]]",
                f"{x.role}",
                f"[link](https://anilist.co/staff/{x.staff.staff_id})",
            ]
            for x in self.staffs
        ]

        writer = MarkdownTableWriter(
            headers=["Image", "Name", "Role", "Link"],
            value_matrix=relations,
            margin=1,
        )

        return writer.dumps()

    def characters_to_md(self) -> str:
        relations = [
            [
                f"![image]({x.character.image})",
                f"[[{sanitize_path(x.character.name)} ({x.character.character_id})|{x.character.name}]]",
                f"{x.role}",
                f"[link](https://anilist.co/character/{x.character.character_id})",
                " <br />".join([f"![image]({va.image})" for va in x.voice_actors]),
                " <br />".join(
                    [
                        f"[[{sanitize_path(va.name)} ({va.staff_id})|{va.name}]]"
                        for va in x.voice_actors
                    ]
                ),
                " <br />".join(
                    [
                        f"[link](https://anilist.co/staff/{va.staff_id})"
                        for va in x.voice_actors
                    ]
                ),
            ]
            for x in self.characters
        ]

        writer = MarkdownTableWriter(
            headers=["Image", "Name", "Role", "Link", "VA Image", "VA", "VA link"],
            value_matrix=relations,
            margin=1,
        )

        return writer.dumps()

    def to_md(self, template: str) -> str:

        return template.format(
            frontmatter=self.to_yaml(),
            user_info=self.user_status.to_md(),
            info=self.info.to_md(),
            studios=self.studios_to_md(),
            relations=self.relations_to_md(),
            staff=self.staff_to_md(),
            description=self.info.description,
            characters=self.characters_to_md(),
            title=self.info.title,
        )


ANIME: Dict[int, Anime] = {}
STAFF_IDS: Set[int] = set()
STUDIO_IDS: Set[int] = set()
CHARACTERS_IDS: Set[int] = set()

url = "https://graphql.anilist.co"

with open("query-anime.graphql", "r", encoding="utf-8") as f:
    USER_ANIME_LIST_QUERY = f.read()


def parse_date(
    year: Optional[int], month: Optional[int], day: Optional[int]
) -> Optional[date]:
    if not year:
        return None

    month = month or 1
    day = day or 1

    return date(year, month, day)


def sanitize_path(path: str) -> str:
    return path.replace("/", "_").replace("~", "_").replace(":", "_")


def query_user_anime_list(username: str) -> Iterator[Anime]:
    variables = {
        "userName": username,
        "chunk": 1,
        "perChunk": 100,
    }

    while True:
        response = requests.post(
            url, json={"query": USER_ANIME_LIST_QUERY, "variables": variables}
        )

        parsed_text = json.loads(response.text)
        parsed_text = parsed_text["data"]["MediaListCollection"]
        has_next_chunk = parsed_text["hasNextChunk"]

        for anime_list in parsed_text["lists"]:
            for entry in anime_list["entries"]:
                if entry["mediaId"] in ANIME:
                    continue

                raw = entry["media"]

                anime = Anime.from_graphql(entry)

                ANIME[anime.info.anilist_id] = anime
                yield anime

        if not has_next_chunk:
            break

        assert type(variables["chunk"]) == int

        variables["chunk"] += 1


def to_markdown(anime: Anime, template: str, root: Path):
    string = anime.to_md(template)

    with open(
        root.joinpath(
            f"{sanitize_path(anime.info.title)} ({anime.info.anilist_id}).md"
        ),
        "w",
    ) as f:
        f.write(string)


def main(root: Path, username: str):
    root.mkdir(parents=True, exist_ok=True)
    template = ""

    with open("template-anime.md", "r") as f:
        template = f.read()

    for i, anime in enumerate(query_user_anime_list(username)):
        print(f"\r\033[K{i: 4} {anime.info.title}", end="")
        to_markdown(anime, template, root)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process some integers.")
    parser.add_argument(
        "username", metavar="USERNAME", type=Path, help="Anilist username"
    )
    parser.add_argument(
        "dst", metavar="PATH", type=Path, help="Path to destination folder"
    )

    args = parser.parse_args()
    parsed_args = vars(args)

    main(**parsed_args)

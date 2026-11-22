import json

from vc.sources.chat_kick import parse_event
from vc.sources.chat_twitch import ChatHub, ChatMessage, parse_irc


def test_parse_irc():
    tagged = "@badge-info=;color=#FF0000;display-name=FooBar;id=1 :foobar!foobar@foobar.tmi.twitch.tv PRIVMSG #Streamer_En :NO WAY KEKW"
    assert parse_irc(tagged)[:3] == ("streamer_en", "foobar", "NO WAY KEKW")
    assert parse_irc(":bob!bob@bob.tmi.twitch.tv PRIVMSG #streamer_x :clip it : now") == ("streamer_x", "bob", "clip it : now", frozenset())
    assert parse_irc(":bob!bob@bob.tmi.twitch.tv PRIVMSG #streamer_x :\x01ACTION waves\x01")[:3] == ("streamer_x", "bob", "waves")
    em = "@emotes=25:0-4,12-16/354:6-10;display-name=A :a!a@a.tmi.twitch.tv PRIVMSG #streamer_x :Kappa Keepo Kappa"
    assert parse_irc(em)[3] == frozenset({"Kappa", "Keepo"})
    assert parse_irc("PING :tmi.twitch.tv") is None
    assert parse_irc(":tmi.twitch.tv 001 justinfan123 :Welcome") is None
    assert parse_irc("@msg-id=sub :tmi.twitch.tv USERNOTICE #streamer_x :hi") is None


def test_ring_buffer_trims(monkeypatch):
    from vc.sources import chat_twitch
    monkeypatch.setattr(chat_twitch, "get_config", lambda: type("C", (), {"chat": type("X", (), {"log": False})})())
    hub = ChatHub()
    seen = []
    hub.listeners.append(seen.append)
    for t in range(0, 1300, 10):
        hub.publish(ChatMessage(1000.0 + t, "streamer_x", "u", "m"))
    buf = hub.buffers["streamer_x"]
    assert buf[-1].ts - buf[0].ts <= 600 and len(seen) == 130
    assert len(hub.recent("streamer_x", 30, now=2290.0)) == 4


def test_kick_parse():
    raw = json.dumps({"event": r"App\Events\ChatMessageEvent", "channel": "chatrooms.42.v2",
                      "data": json.dumps({"content": "W", "sender": {"username": "Bob"}})})
    assert parse_event(raw) == ("bob", "W")
    assert parse_event(json.dumps({"event": "pusher:ping"})) is None

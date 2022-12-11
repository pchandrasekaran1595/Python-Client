import sys
import requests


def main():

    args_1: str = "--salary"
    args_2: str = "--from"
    args_3: str = "--to"
    args_4: str = "--base-url"

    salary: float = 0.0
    from_country: str = "QAT"
    to_country: str = "IND"
    base_url: str = "http://127.0.0.1:50002"

    if args_1 in sys.argv: salary = float(sys.argv[sys.argv.index(args_1) + 1]) 
    if args_2 in sys.argv: from_country = sys.argv[sys.argv.index(args_2) + 1]
    if args_3 in sys.argv: to_country = sys.argv[sys.argv.index(args_3) + 1]
    if args_4 in sys.argv: base_url = sys.argv[sys.argv.index(args_4) + 1]

    payload = {
        "salary" : salary,
        "from_country" : from_country,
        "to_country" : to_country
    }

    response = requests.request(method="POST", url=f"{base_url}/convert", json=payload)
    print(response.json())


if __name__ == "__main__":
    sys.exit(main() or 0)
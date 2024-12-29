from src.train import train_agent
from src.test import test_agent
from src.visualisation import test_agent_obs

if __name__ == "__main__":
    print("1 -> Train Agent")
    print("2 -> Test Agent")
    print("3 -> Show Test Example")
    choice = input("Choose an option:")

    if choice == "1":
        train_agent()
    elif choice == "2":
        test_agent()
    elif choice == "3":
        test_agent_obs()
    else:
        print("Invalid choice.")

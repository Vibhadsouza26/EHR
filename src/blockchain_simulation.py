# File: src/blockchain_simulation.py

import hashlib
import time

class Block:
    def __init__(self, index: int, data: str, previous_hash: str):
        self.index = index
        self.timestamp = time.time()
        self.data = data  # In a real EHR system, this could be a record or its hash
        self.previous_hash = previous_hash
        self.hash = self.compute_hash()
    
    def compute_hash(self) -> str:
        block_contents = f"{self.index}{self.timestamp}{self.data}{self.previous_hash}"
        return hashlib.sha256(block_contents.encode()).hexdigest()

class Blockchain:
    def __init__(self):
        self.chain = [self.create_genesis_block()]
    
    def create_genesis_block(self) -> Block:
        return Block(0, "Genesis Block", "0")
    
    def add_block(self, data: str):
        previous_block = self.chain[-1]
        new_block = Block(len(self.chain), data, previous_block.hash)
        self.chain.append(new_block)
    
    def is_chain_valid(self) -> bool:
        for i in range(1, len(self.chain)):
            current = self.chain[i]
            previous = self.chain[i - 1]
            if current.hash != current.compute_hash():
                return False
            if current.previous_hash != previous.hash:
                return False
        return True

if __name__ == "__main__":
    # Simulate storing a hash of a sample EHR record
    sample_ehr = "PatientID:12345|Diagnosis:Hypertension|Medications:Amlodipine"
    ehr_hash = hashlib.sha256(sample_ehr.encode()).hexdigest()
    
    # Initialize the blockchain and add a block with the EHR hash
    ehr_blockchain = Blockchain()
    ehr_blockchain.add_block(ehr_hash)
    
    # Validate and print the blockchain
    print("Is the blockchain valid?", ehr_blockchain.is_chain_valid())
    for block in ehr_blockchain.chain:
        print(f"Block {block.index} - Hash: {block.hash}")

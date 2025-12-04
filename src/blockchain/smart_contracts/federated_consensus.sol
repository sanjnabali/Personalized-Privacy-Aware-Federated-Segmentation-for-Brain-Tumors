// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

contract FederatedConsensus {
    struct Update {
        string clientId;
        string ipfsCid;
        uint256 timestamp;
    }

    mapping(uint256 => Update[]) public roundUpdates;
    string public globalModelCid;
    uint256 public currentRound;

    event UpdateSubmitted(string indexed clientId, uint256 round, string cid);
    
    // Updated Constructor: Simple init
    constructor() {
        currentRound = 0;
        globalModelCid = "GENESIS_BLOCK";
    }

    function submitUpdate(string memory _clientId, string memory _ipfsCid) public {
        roundUpdates[currentRound].push(Update(_clientId, _ipfsCid, block.timestamp));
        emit UpdateSubmitted(_clientId, currentRound, _ipfsCid);
    }
}
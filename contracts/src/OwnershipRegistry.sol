// SPDX-License-Identifier: MIT
pragma solidity ^0.8.24;

import {Ownable} from "@openzeppelin/contracts/access/Ownable.sol";

/// @title OwnershipRegistry
/// @notice Anchors content-hash ownership proofs on-chain for DONTAI artworks.
///         Only a content hash + owner + timestamp are stored on-chain; the
///         actual image, metadata, and C2PA manifest live off-chain (see
///         PROJECT_DESIGN.md §5-1). This keeps registration cheap and avoids
///         putting any personal/creative content on a public chain.
/// @dev Design reference: PROJECT_DESIGN.md §5-3.
contract OwnershipRegistry is Ownable {
    struct Record {
        address owner;
        bytes32 contentHash;
        uint64 timestamp;
        bool doNotTrain;
    }

    /// @dev tokenId => Record. tokenId 0 is never assigned, so it doubles as "not found".
    mapping(uint256 => Record) public records;

    /// @dev contentHash => tokenId, for O(1) duplicate checks and lookup by hash.
    mapping(bytes32 => uint256) public hashToToken;

    /// @dev Addresses allowed to call `registerFor` on behalf of a creator
    ///      (gas-sponsored registration via the platform relayer — see
    ///      PROJECT_DESIGN.md §5-4). The relayer's own key is held in the KMS,
    ///      not in application code.
    mapping(address => bool) public relayers;

    uint256 public nextId;

    event Registered(uint256 indexed id, address indexed owner, bytes32 contentHash, bool doNotTrain);
    event Transferred(uint256 indexed id, address indexed from, address indexed to);
    event RelayerUpdated(address indexed relayer, bool allowed);

    error AlreadyRegistered(bytes32 contentHash);
    error NotFound(uint256 id);
    error NotOwner(uint256 id, address caller);
    error NotRelayer(address caller);
    error ZeroAddress();

    constructor(address initialOwner) Ownable(initialOwner) {}

    modifier onlyRelayer() {
        if (!relayers[msg.sender]) revert NotRelayer(msg.sender);
        _;
    }

    /// @notice Self-service registration: caller pays their own gas and becomes the owner.
    function register(bytes32 contentHash, bool doNotTrain) external returns (uint256 id) {
        id = _register(msg.sender, contentHash, doNotTrain);
    }

    /// @notice Relayer-sponsored registration: the platform relayer submits the
    ///         transaction (and pays gas) on behalf of `owner`, who never needs
    ///         their own gas balance. Restricted to addresses the contract
    ///         owner has explicitly authorized via `setRelayer`.
    function registerFor(address owner, bytes32 contentHash, bool doNotTrain)
        external
        onlyRelayer
        returns (uint256 id)
    {
        if (owner == address(0)) revert ZeroAddress();
        id = _register(owner, contentHash, doNotTrain);
    }

    function _register(address owner, bytes32 contentHash, bool doNotTrain) private returns (uint256 id) {
        if (hashToToken[contentHash] != 0) revert AlreadyRegistered(contentHash);

        id = ++nextId;
        records[id] = Record({owner: owner, contentHash: contentHash, timestamp: uint64(block.timestamp), doNotTrain: doNotTrain});
        hashToToken[contentHash] = id;

        emit Registered(id, owner, contentHash, doNotTrain);
    }

    /// @notice Look up a registration by content hash (e.g. recomputed perceptual hash
    ///         of a suspected infringing image) to prove prior registration.
    function verify(bytes32 contentHash)
        external
        view
        returns (bool exists, address owner, uint64 timestamp, bool doNotTrain)
    {
        uint256 id = hashToToken[contentHash];
        if (id == 0) return (false, address(0), 0, false);

        Record storage r = records[id];
        return (true, r.owner, r.timestamp, r.doNotTrain);
    }

    /// @notice Transfer ownership of a registered record. Only the current
    ///         on-chain owner may call this directly (no relayer path for
    ///         transfers in the MVP — see PROJECT_DESIGN.md §5-3 note on
    ///         upgrading to ERC-721 if marketplace transfers are needed later).
    function transfer(uint256 id, address to) external {
        Record storage r = records[id];
        if (r.owner == address(0)) revert NotFound(id);
        if (r.owner != msg.sender) revert NotOwner(id, msg.sender);
        if (to == address(0)) revert ZeroAddress();

        r.owner = to;
        emit Transferred(id, msg.sender, to);
    }

    /// @notice Authorize or revoke a relayer address. Only the contract owner
    ///         (platform admin) can call this.
    function setRelayer(address relayer, bool allowed) external onlyOwner {
        if (relayer == address(0)) revert ZeroAddress();
        relayers[relayer] = allowed;
        emit RelayerUpdated(relayer, allowed);
    }
}

// components/TaskDetails/ActionButtons.tsx
import React from 'react';

interface ActionButtonsProps {
  onConfirm: () => void;
  onReject: () => void;
  submitting: boolean;
}

const ActionButtons: React.FC<ActionButtonsProps> = ({
  onConfirm,
  onReject,
  submitting
}) => {
  return (
    <div className="flex space-x-4">
      <button
        className="flex-1 px-4 py-2 bg-green-600 text-white rounded-md hover:bg-green-700 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
        onClick={onConfirm}
        disabled={submitting}
      >
        {submitting ? 'Submitting...' : 'Confirm Match ✓'}
      </button>
      
      <button
        className="flex-1 px-4 py-2 bg-red-600 text-white rounded-md hover:bg-red-700 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
        onClick={onReject}
        disabled={submitting}
      >
        {submitting ? 'Submitting...' : 'Cut Link ✗'}
      </button>
    </div>
  );
};

export default ActionButtons;